from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

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
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.temp = nn.Parameter(0.07*torch.ones([]))  
        self.temp_token = nn.Parameter(0.07*torch.ones([])) 
        self.similarity_threshold = 0.2 
        
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
        
        l_token_embed = text_output.last_hidden_state[:,1:,:]
        language_mask = text.attention_mask[:,1:]
        v_patch_embed = image_embeds[:,1:,:]
                
        v_embed = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        l_embed = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)        
        
        ###============== Image-text Contrastive Learning ===================###
        sim_targets = torch.eye(l_embed.shape[0]).to(l_embed.device)
            
        # ----------- Global Loss ----------- #
        sim_i2t = v_embed @ l_embed.t() / self.temp 
        sim_t2i = l_embed @ v_embed.t() / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        # ----------- Local Loss ----------- #
        similarity = torch.einsum('btd,bpd->btp',l_token_embed, v_patch_embed)
        similarity = (similarity-torch.min(similarity, dim=-1).values.unsqueeze(-1))/(torch.max(similarity, dim=-1).values - torch.min(similarity, dim=-1).values).unsqueeze(-1)
        similarity = torch.where(similarity < self.similarity_threshold, 0.0, similarity)
        v_align_weights = similarity/torch.sum(similarity, dim=-1).unsqueeze(-1)

        # B x T x D (per token, linear combination of patch embeddings)
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, dim=-1)

        # token embedding
        l_token_embed = F.normalize(l_token_embed, dim=-1)

        # B x T -> B x T x T
        mask_logits = (1.0-language_mask).unsqueeze(1).expand(-1, l_token_embed.shape[1], -1)
        
        # 1 x T x T -> B x T x T (identity)
        sim_token_targets = torch.eye(l_token_embed.shape[1]).unsqueeze(0).expand(l_token_embed.shape[0], -1, -1).to(similarity.device)

        # B x T x T [sim between patch-aware token rep with original token]
        sim_token_i2t = torch.einsum('bmd,bnd->bmn', l_grouped_v_patch_embed, l_token_embed) / self.temp_token
        sim_token_t2i = torch.einsum('bmd,bnd->bmn', l_token_embed, l_grouped_v_patch_embed) / self.temp_token

        mask_logits = mask_logits.bool()

        # True indicates unpadded
        # B x T
        unpadded_1d_token_mask = language_mask.bool()

        # B x T x T
        unpadded_2d_token_mask = unpadded_1d_token_mask.unsqueeze(-1) * unpadded_1d_token_mask.unsqueeze(1)
        
        # B x T x T
        padded_2d_token_mask = ~unpadded_2d_token_mask

        # B x [padded row token] x [padded col token] will have uniform distribution (since all 0s)
        # but we remove then later anyways
        min_value = torch.finfo(sim_token_i2t.dtype).min

        sim_token_i2t = sim_token_i2t.masked_fill(padded_2d_token_mask, min_value)
        sim_token_t2i = sim_token_t2i.masked_fill(padded_2d_token_mask, min_value)
        sim_token_i2t = torch.flatten(sim_token_i2t, start_dim=0, end_dim=1)
        sim_token_t2i = torch.flatten(sim_token_t2i, start_dim=0, end_dim=1)
        sim_token_targets = torch.flatten(sim_token_targets, start_dim=0, end_dim=1)
        
        # B x T x T
        log_prob_i2t = F.log_softmax(sim_token_i2t, dim=-1)
        log_prob_t2i = F.log_softmax(sim_token_t2i, dim=-1)

        # B x T
        # loss per token
        loss_token_i2t = (-log_prob_i2t*sim_token_targets)#.sum(-1)
        loss_token_t2i = (-log_prob_t2i*sim_token_targets)#.sum(-1)

        unpadded_2d_token_mask_flattened = torch.flatten(unpadded_2d_token_mask, start_dim=0, end_dim=1)
        loss_token_i2t = loss_token_i2t[unpadded_2d_token_mask_flattened]
        loss_token_t2i = loss_token_t2i[unpadded_2d_token_mask_flattened]

        # remove padded tokens from loss
        # loss_token_i2t = loss_token_i2t[unpadded_1d_token_mask]
        # loss_token_t2i = loss_token_t2i[unpadded_1d_token_mask]
        

        # taken mean over all tokens 
        # [slightly different from mean over token then mean over batch]
        # but I feel this is better loss as number of tokens vary within batch
        loss_token_i2t = loss_i2t.mean()
        loss_token_t2i = loss_token_t2i.mean()

        loss_token_ita = 0.5*(loss_token_i2t + loss_token_t2i)       

        return loss_ita, loss_token_ita
 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
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