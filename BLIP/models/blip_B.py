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
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.entries = 20
        self.domain_dim = embed_dim
        
        self.codebooks = nn.Embedding(self.entries, self.domain_dim)
        self.domain_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.codebooks_m = nn.Embedding(self.entries, self.domain_dim)
        self.domain_attention_m = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                            [self.codebooks, self.codebooks_m],
                            [self.domain_attention,self.domain_attention_m],
                           ] 
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("image_queue_u", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue_u", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))  
        
        self.negative_all_rank = negative_all_rank
        
        
    def forward(self, image, caption, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        input_tokens = torch.arange(0, self.entries).unsqueeze(0).expand(len(image), -1).to(image.device)
        domain_rep = self.codebooks(input_tokens)
        
        image_embeds = self.visual_encoder(image) 
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
                
        v_embed_u = self.vision_proj(image_embeds[:,0,:])   
        v_embed_a,_ = self.domain_attention(v_embed_u.unsqueeze(1), domain_rep, domain_rep) 
        v_embed = F.normalize(v_embed_a.squeeze(1), dim=-1)
        v_embed_cls = F.normalize(v_embed_u, dim=-1)
        l_embed_u = self.text_proj(text_output.last_hidden_state[:,0,:])
        l_embed_a,_ = self.domain_attention(l_embed_u.unsqueeze(1), domain_rep, domain_rep)
        l_embed = F.normalize(l_embed_a.squeeze(1), dim=-1)        
        l_embed_cls = F.normalize(l_embed_u, dim=-1)
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            v_embed_m_u = self.vision_proj_m(image_embeds_m[:,0,:]) 
            v_embed_m_a,_ = self.domain_attention_m(v_embed_m_u.unsqueeze(1), domain_rep, domain_rep) 
            v_embed_m = F.normalize(v_embed_m_a.squeeze(1), dim=-1)
            v_embed_m_all = torch.cat([v_embed_m.t(),self.image_queue.clone().detach()],dim=1)  
            v_embed_m_all_u = torch.cat([F.normalize(v_embed_m_u, dim=-1).t(),self.image_queue_u.clone().detach()],dim=1)            
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            l_embed_m_u = self.text_proj_m(text_output_m.last_hidden_state[:,0,:])
            l_embed_m_a,_ = self.domain_attention_m(l_embed_m_u.unsqueeze(1), domain_rep, domain_rep) 
            l_embed_m = F.normalize(l_embed_m_a.squeeze(1), dim=-1)
            l_embed_m_all = torch.cat([l_embed_m.t(),self.text_queue.clone().detach()],dim=1)
            l_embed_m_all_u = torch.cat([F.normalize(l_embed_m_u, dim=-1).t(),self.text_queue_u.clone().detach()],dim=1)
            
            # l_token_embed_m = text_output_m.last_hidden_state[:,1:,:]
            # v_patch_embed_m = image_embeds[:,1:,:]       
            
        # ----------- Global Loss ----------- #
        sim_i2t = v_embed @ l_embed_m_all / self.temp 
        sim_t2i = l_embed @ v_embed_m_all / self.temp 
        
        sim_i2t_u = v_embed_u @ l_embed_m_all_u / self.temp_again 
        sim_t2i_u = l_embed_u @ v_embed_m_all_u / self.temp_again 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 
        loss_i2t_u = -torch.sum(F.log_softmax(sim_i2t_u, dim=1)*sim_targets,dim=1).mean()
        loss_t2i_u = -torch.sum(F.log_softmax(sim_t2i_u, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        loss_ita_u = (loss_i2t_u+loss_t2i_u)/2
        
        loss = 0.4*loss_ita + 0.6*loss_ita_u
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(v_embed_m, l_embed_m, v_embed_m_u, l_embed_m_u, idxs)

        return loss
 

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
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_feats_u, text_feats_u, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.image_queue_u[:, ptr:ptr + batch_size] = image_feats_u.T
        self.text_queue_u[:, ptr:ptr + batch_size] = text_feats_u.T
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