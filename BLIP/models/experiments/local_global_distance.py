from models.med import BertConfig, BertModel
from transformers import BertTokenizer
from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.experiments.alignment_module import multi_alignment_block

from models.blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        
        # frozen encoders
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        
        # trainable encoders
        self.visual_encoder_new, vision_width_new = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.text_encoder_new = BertModel(config=med_config, add_pooling_layer=False)  
        
        text_width = self.text_encoder.config.hidden_size
        med_config.encoder_width = vision_width
        
        # frozen proojection layers
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        
        # trainable projection layers
        self.vision_proj_new = nn.Linear(vision_width, embed_dim)
        self.text_proj_new = nn.Linear(text_width, embed_dim)
        self.v_patch_proj = nn.Linear(vision_width, embed_dim)
        self.l_word_proj = nn.Linear(text_width, embed_dim)
        
        # create momentum encoders and projection layers
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.v_patch_proj_m = nn.Linear(vision_width, embed_dim)
        self.l_word_proj_m = nn.Linear(text_width, embed_dim)     
        
        # pool patches
        self.pool_patch = nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4)
        self.pool_patch_m = nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4)
        
        # codebooks
        self.entries = 1000
        self.v_codebook = nn.Embedding(self.entries, embed_dim)
        self.l_codebook = nn.Embedding(self.entries, embed_dim)
        self.v_codebook_m = nn.Embedding(self.entries, embed_dim)
        self.l_codebook_m = nn.Embedding(self.entries, embed_dim)
        self.codebook_attention = multi_alignment_block(embed_dim, 6)
        self.codebook_attention_m = multi_alignment_block(embed_dim, 6)
          
        # freeze the pretrained encoders and initialize the new encoders    
        self.model_pairs_pretrained = [[self.visual_encoder_new, self.visual_encoder],
                                       [self.vision_proj_new, self.vision_proj],
                                       [self.text_encoder_new, self.text_encoder],
                                       [self.text_proj_new, self.text_proj]]
        #self.initialize()
        
        # freeze the paramters of momentum encoder
        self.model_pairs = [[self.visual_encoder_new,self.visual_encoder_m],
                            [self.vision_proj_new,self.vision_proj_m],
                            [self.text_encoder_new,self.text_encoder_m],
                            [self.text_proj_new,self.text_proj_m],
                            [self.v_codebook,self.v_codebook_m],
                            [self.l_codebook,self.l_codebook_m],
                            [self.codebook_attention, self.codebook_attention_m],
                            [self.v_patch_proj, self.v_patch_proj_m],
                            [self.l_word_proj, self.l_word_proj_m],
                            [self.pool_patch, self.pool_patch_m]
                           ] 
        #self.copy_params()
        
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("image_patch_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_word_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        self.image_patch_queue = F.normalize(self.image_patch_queue, dim=0)
        self.text_word_queue = F.normalize(self.text_word_queue, dim=0)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))  
        self.fine_temp = nn.Parameter(0.007*torch.ones([]))
        self.local_temp = nn.Parameter(0.007*torch.ones([]))
        self.similarity_threshold = 0.06
        self.momentum=False
        
        
    def forward(self, image, caption, idx, topic):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            self.fine_temp.clamp_(0.0001,0.1)
            
        v_cls_frozen, l_cls_frozen = self.get_frozen_feat(image, caption)
        v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask = self.get_trainable_feat(image, caption, momentum=False)
        
        ###============== Image-text Contrastive Learning ===================###
        if self.momentum:
            idx = idx.view(-1,1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
            pos_idx = torch.eq(idx, idx_all).float()       
            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
            
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                v_patch_cls_m, l_word_cls_m, v_cls_m, l_cls_m, language_mask = self.get_trainable_feat(image, caption, momentum=True)
                
                v_cls_m_all = torch.cat([v_cls_m.t(),self.image_queue.clone().detach()],dim=1)                   
                l_cls_m_all = torch.cat([l_cls_m.t(),self.text_queue.clone().detach()],dim=1) 
                v_patch_cls_m_all = torch.cat([v_patch_cls_m.t(),self.image_patch_queue.clone().detach()],dim=1)
                l_word_cls_m_all = torch.cat([l_word_cls_m.t(),self.text_word_queue.clone().detach()],dim=1) 
                
        else:
            sim_targets = torch.eye(v_patch.shape[0]).to(v_patch.device)
            
        # ---------- Local similarity matrix -------- #    
        # similarity calculation BxTxP
        similarity = torch.einsum('btd,bpd->btp',l_word, v_patch)
        
        # min-max normalisation 
        eps = 1e-6
        similarity = (similarity-torch.min(similarity, dim=-1).values.unsqueeze(-1))/(torch.max(
            similarity, dim=-1).values.unsqueeze(-1) - torch.min(similarity, dim=-1).values.unsqueeze(-1) + eps)
        
        # thresholding
        similarity = torch.where(similarity < self.similarity_threshold, 0.0, similarity)
        
        print(torch.max(similarity[0], dim=-1))
        print(caption[0])
        print(topic[0])
        exit()
        print("-"*20)
        
        # alignment-weighting
        v_align_weights = similarity/torch.sum(similarity, dim=-1).unsqueeze(-1)
        v_patch = torch.einsum('btp,bpd->btd', v_align_weights, v_patch)
        
        # local loss
        local_vl = self.local_pairwise_contrastive_loss(v_patch, l_word, language_mask)
        local_lv = self.local_pairwise_contrastive_loss(l_word, v_patch, language_mask)
        
        # ----------- Global Similarity matrices ----------- #
        if self.momentum:
            cls_sim_i2t = v_cls @ l_cls_m_all / self.temp 
            cls_sim_t2i = l_cls @ v_cls_m_all / self.temp
            
            pooled_sim_i2t = v_patch_cls @ l_word_cls_m_all / self.fine_temp
            pooled_sim_t2i = l_word_cls @ v_patch_cls_m_all / self.fine_temp
            
        else:
            cls_sim_i2t = v_cls @ l_cls.t() / self.temp 
            cls_sim_t2i = l_cls @ v_cls.t() / self.temp
            
            pooled_sim_i2t = v_patch_cls @ l_word_cls.t() / self.fine_temp
            pooled_sim_t2i = l_word_cls @ v_patch_cls.t() / self.fine_temp
        
        # ----------- Losses -------------- #
        cls_loss_i2t = -torch.sum(F.log_softmax(cls_sim_i2t, dim=1)*sim_targets,dim=1).mean()
        cls_loss_t2i = -torch.sum(F.log_softmax(cls_sim_t2i, dim=1)*sim_targets,dim=1).mean()
        
        pooled_loss_i2t = -torch.sum(F.log_softmax(pooled_sim_i2t, dim=1)*sim_targets, dim=-1).mean()
        pooled_loss_t2i = -torch.sum(F.log_softmax(pooled_sim_t2i, dim=1)*sim_targets, dim=-1).mean()
        
        distance_v = (1-torch.sum(v_cls_frozen*v_cls, dim=-1)).mean()
        distance_l = (1-torch.sum(l_cls_frozen*l_cls, dim=-1)).mean()
        
        l1 = 0.5*(cls_loss_i2t + cls_loss_t2i)
        l2_global = 10*(pooled_loss_i2t + pooled_loss_t2i)
        l2_local = 25*(local_lv + local_vl)
        l2 = 0.5*(l2_global + l2_local)
        l3 = 0.5*(distance_l + distance_v)
        
        #loss = (l1 + l2)/2.0
        #loss = l2_global
        loss = l2_local
        
        if self.momentum:
            idxs = concat_all_gather(idx)
            self._dequeue_and_enqueue(v_cls_m, l_cls_m, v_patch_cls_m, l_word_cls_m, idxs)   
        
        # TODO: For local loss
        # 1. Pool patches to make it 32 or 16, until we see tangible visual evidence per patch
        # 2. Examine attention on chosen images till we identify a suitable setting (set temperature forcing attention to be peaky?) 
        # 3. For later: Could we use weak supervision from object detectors to initialize the patch-token attention distributions?
        loss_dict = {
            "loss":loss,
            "L1":l1,
            "L2":l2,
            "L3":l3,
            "L2_local":l2_local,
            "L2_global":l2_global
        }     

        return loss_dict
    
    def get_frozen_feat(self, i, t):
        # TODO: Do we need to return language mask as well 
        # image cls
        image_last_frozen, image_patch_frozen = self.visual_encoder(i, return_layer='patch_and_last')
        v_cls_frozen = F.normalize(self.vision_proj(image_last_frozen[:,0,:]),dim=-1)
        
        # text cls
        text = self.tokenizer(t, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(i.device)
        text_output_frozen, word_embed_frozen = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
        l_cls_frozen = F.normalize(self.text_proj(text_output_frozen.last_hidden_state[:,0,:]),dim=-1)
        
        return v_cls_frozen, l_cls_frozen
                
    
    def get_trainable_feat(self, i, t, momentum=False):
        code_tokens = torch.arange(0, self.entries).unsqueeze(0).expand(i.shape[0], -1).to(i.device)
        text = self.tokenizer(t, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(i.device)
        if not momentum:
            # patch embeddings from trainable image encoder
            image_last, image_patch = self.visual_encoder_new(i, return_layer='patch_and_last')
            #v_patch = self.v_patch_proj(image_patch)
            v_patch = self.v_patch_proj(image_last[:,1:,:])
        
            # pool the image patches
            B,T,D = v_patch.shape
            v_patch = v_patch.view(B,16,16,D).permute(0,3,1,2)
            v_patch = self.pool_patch(v_patch).permute(0,2,3,1).view(B,16,D)
        
            # word embeddings from trainable text encoder
            text_output, word_embed = self.text_encoder_new(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            language_mask = text.attention_mask[:,1:]
            l_word = self.l_word_proj(word_embed[:,1:,:])
        
            # cls from trainable encoders
            v_cls = F.normalize(self.vision_proj_new(image_last[:,0,:]),dim=-1)
            l_cls = F.normalize(self.text_proj_new(text_output.last_hidden_state[:,0,:]),dim=-1)
            
            # pooled cls after attention over codebooks for new patch and word embeddings
            v_codes = self.v_codebook(code_tokens)
            l_codes = self.l_codebook(code_tokens)
            v_patch = self.codebook_attention(v_patch, v_codes)
            l_word = self.codebook_attention(l_word, l_codes)
            v_patch_cls = F.normalize(torch.mean(v_patch, dim=1), dim=-1)
            l_word_cls = F.normalize(torch.mean(l_word, dim=1), dim=-1)
            
            return v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask
        else:
            # patch embeddings from trainable image encoder
            image_last_m, image_patch_m = self.visual_encoder_m(i, return_layer='patch_and_last')
            v_patch_m = self.v_patch_proj_m(image_patch_m)
        
            # pool the image patches
            B,T,D = v_patch_m.shape
            v_patch_m = v_patch_m.view(B,16,16,D).permute(0,3,1,2)
            v_patch_m = self.pool_patch_m(v_patch_m).permute(0,2,3,1).view(B,16,D)
        
            # word embeddings from trainable text encoder
            text_output_m, word_embed_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            language_mask = text.attention_mask[:,1:]
            l_word_m = self.l_word_proj_m(word_embed_m[:,1:,:])
        
            # cls from trainable encoders
            v_cls_m = F.normalize(self.vision_proj_m(image_last_m[:,0,:]),dim=-1)
            l_cls_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
            
            # pooled cls after attention over codebooks for new patch and word embeddings
            v_codes_m = self.v_codebook_m(code_tokens)
            l_codes_m = self.l_codebook_m(code_tokens)
            v_patch_m = self.codebook_attention_m(v_patch_m, v_codes_m)
            l_word_m = self.codebook_attention_m(l_word_m, l_codes_m)
            v_patch_cls_m = F.normalize(torch.mean(v_patch_m, dim=1), dim=-1)
            l_word_cls_m = F.normalize(torch.mean(l_word_m, dim=1), dim=-1)
            
            return v_patch_cls_m, l_word_cls_m, v_cls_m, l_cls_m, language_mask
            
            
    
    def local_pairwise_contrastive_loss(self, a, b, mask):
        batch_size, seq_len, _ = a.shape
        mask_logits = ~(mask.bool().unsqueeze(-1) * mask.bool().unsqueeze(1))
        labels = torch.eye(seq_len).repeat(batch_size,1,1).to(a.device)
        logits = torch.einsum('bmd,bnd->bmn',a,b)/self.local_temp
        INF = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(mask_logits, INF)
        
        # flatten all vectors
        logits = torch.flatten(logits, 0, 1)
        labels = torch.flatten(labels, 0, 1)
        mask_logits = torch.flatten(mask_logits, 0, 1)
        
        # loss
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -log_prob*labels
        loss = loss.masked_fill(~mask_logits, 0)
        loss = torch.sum(loss, dim=-1).mean()
        return loss
    
    @torch.no_grad()
    def inference(self, image, caption):
        v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask = self.get_trainable_feat(image, caption, momentum=False)
        
        return v_cls, l_cls, v_patch_cls, l_word_cls
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
                
    @torch.no_grad()
    def initialize(self):
        for model_pair in self.model_pairs_pretrained:
            for param, param_pre in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param.data.copy_(param_pre.data) # initialize
                param_pre.requires_grad = False # not update by gradient

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_fine_feat, text_fine_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        image_fine_feats = concat_all_gather(image_fine_feat)
        text_fine_feats = concat_all_gather(text_fine_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.image_patch_queue[:, ptr:ptr + batch_size] = image_fine_feats.T
        self.text_word_queue[:, ptr:ptr + batch_size] = text_fine_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        #print(msg.missing_keys)
    model.initialize()
    model.copy_params()
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #     for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # output = torch.cat(tensors_gather, dim=0)
    output = tensor
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
