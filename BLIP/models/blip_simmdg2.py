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

        self.vision_proj_1 = nn.Linear(embed_dim //2, embed_dim//2)
        self.text_proj_1 = nn.Linear(embed_dim//2, embed_dim//2)
        self.vision_proj_1_m = nn.Linear(embed_dim //2, embed_dim//2)
        self.text_proj_1_m = nn.Linear(embed_dim//2, embed_dim//2)

        self.cross_modal_text_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_modal_img_proj = nn.Linear(embed_dim, embed_dim)


        # self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                            [self.vision_proj_1, self.vision_proj_1_m],
                            [self.text_proj_1, self.text_proj_1_m]
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim//2, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim//2, queue_size))
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
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = self.vision_proj(image_embeds[:,0,:])

        image_feat_shared = image_feat[:,:int(image_feat.shape[1]/2)]
        image_feat_specific = image_feat[:,int(image_feat.shape[1]/2):]

        image_feat_shared = self.vision_proj_1(image_feat_shared)
  
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = self.text_proj(text_output.last_hidden_state[:,0,:])

        text_feat_shared = text_feat[:,:int(text_feat.shape[1]/2)]
        text_feat_specific = text_feat[:,int(text_feat.shape[1]/2):]
        
        text_feat_shared = self.text_proj_1(text_feat_shared)


        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  

            image_feat_m_shared = image_feat_m[:,:int(image_feat_m.shape[1]/2)]
            image_feat_m_shared = self.vision_proj_1_m(image_feat_m_shared)

            image_feat_m_all = torch.cat([image_feat_m_shared.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 

            text_feat_m_shared = text_feat_m[:,:int(text_feat_m.shape[1]/2)]
            text_feat_m_shared = self.text_proj_1_m(text_feat_m_shared)

            text_feat_m_all = torch.cat([text_feat_m_shared.t(),self.text_queue.clone().detach()],dim=1)

            # sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            # sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            # sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        image_feat_n = F.normalize(image_feat_shared, dim = -1)
        text_feat_n = F.normalize(text_feat_shared, dim = -1)
        image_feat_m_all_n = F.normalize(image_feat_m_all, dim = -1)
        text_feat_m_all_n = F.normalize(text_feat_m_all, dim = -1)

        sim_i2t = image_feat_n @ text_feat_m_all_n / self.temp 
        sim_t2i = text_feat_n @ image_feat_m_all_n / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m_shared, text_feat_m_shared, idxs)        

        distance_loss_image = -(torch.norm(image_feat_shared - image_feat_specific, dim = -1)).mean()
        distance_loss_text = -(tyorch.norm(text_feat_shared - text_feat_specific, dim = -1)).mean()

        distance_loss = (distance_loss_image + distance_loss_text)/2

        text_proj_to_img = self.cross_modal_text_proj(text_feat)
        img_proj_to_text = self.cross_modal_img_proj(image_feat)

        cross_modal_loss_image = (torch.norm(text_proj_to_img - image_feat, dim = -1)).mean()
        cross_modal_loss_text = (torch.norm(img_proj_to_text - text_feat, dim = -1)).mean()

        cross_modal_loss = (cross_modal_loss_text + cross_modal_loss_image)/2

        # TODO: Might need to tweak these scaling factors
        return 3*loss_ita + 0.1*distance_loss + 0.03*cross_modal_loss


 

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