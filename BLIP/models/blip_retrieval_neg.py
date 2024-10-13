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

        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ] 
        self.copy_params()

        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank
        self.label_smoothing = 0
        self.max_text_tokens = 35
        
        
    def forward(self, image, caption, alpha, idx, neg_images, neg_captions):
        """blip with negatives

        Args:
            image (_type_): B x img (pos)
            caption (_type_): _description_ (pos)
            alpha (_type_): _description_
            idx (_type_): _description_
            neg_images (_type_): B x N x img (N is no. of negatives)
            neg_captions (_type_): List[ List[str]] -> len(List[i]) == N

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_text_tokens, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')       
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  
        
        ###============== Image-text Contrastive Learning ===================###
        
        #Negativ images
        B, N, channel, width, height = neg_images.shape
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            neg_image_embeds_m = self.visual_encoder_m(neg_images.view(B*N, channel, width, height)) 
            pos_image_embeds_m = self.visual_encoder_m(image)
            neg_image_feat_m = F.normalize(self.vision_proj_m(neg_image_embeds_m[:,0,:]),dim=-1)  
            pos_image_feat_m = F.normalize(self.vision_proj_m(pos_image_embeds_m[:,0,:]),dim=-1)
            _, D = neg_image_feat_m.shape
            neg_image_feat_m = neg_image_feat_m.view(B, N, D)
            
            neg_captions = [sent for sentences in neg_captions for sent in sentences]
            neg_tokens = self.tokenizer(neg_captions, padding='max_length', truncation=True, max_length=self.max_text_tokens, 
                              return_tensors="pt").to(image.device) 
            neg_text_output_m = self.text_encoder_m(neg_tokens.input_ids, attention_mask = neg_tokens.attention_mask,                      
                                        return_dict = True, mode = 'text')
            pos_text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
            neg_text_feat_m = F.normalize(self.text_proj_m(neg_text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            pos_text_feat_m = F.normalize(self.text_proj_m(pos_text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            neg_text_feat_m = neg_text_feat_m.view(B, N, D)
            
        pos_i2t = (image_feat * pos_text_feat_m).sum(-1).unsqueeze(-1)
        pos_t2i = (text_feat * pos_image_feat_m).sum(-1).unsqueeze(-1)
        neg_i2t = image_feat.unsqueeze(1) @ neg_text_feat_m.permute(0, 2, 1)
        neg_t2i = text_feat.unsqueeze(1) @ neg_image_feat_m.permute(0, 2, 1)
        neg_i2t = neg_i2t.squeeze(1)
        neg_t2i = neg_t2i.squeeze(1)
        sim_i2t = torch.cat([pos_i2t, neg_i2t], dim=1) / self.temp
        sim_t2i = torch.cat([pos_t2i, neg_t2i], dim=1) / self.temp
        
        with torch.no_grad():
            true_dist = torch.zeros_like(sim_i2t)
            true_dist.fill_(self.label_smoothing / N)
            true_dist[:, 0] = true_dist[:, 0] + 1.0 - self.label_smoothing
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*true_dist,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*true_dist,dim=1).mean()    
        
        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )  
    
        with torch.no_grad():                
            weights_i2t = F.softmax(neg_i2t,dim=1) / self.temp
            weights_t2i = F.softmax(neg_t2i,dim=1) / self.temp  

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])            
            
        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                         
          

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)    

        return loss_i2t, loss_t2i, loss_itm 
 

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
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
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
