from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint

@torch.no_grad()
def freeze_params(model):
    for params in model.parameters():
        params.requires_grad = False

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
                 label_smoothing = 0.,
                 temperature_damp = False,
                 damp_factor = 0.98,
                 warmup_step = 100,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)

        freeze_params(self.visual_encoder)

        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
                  
        freeze_params(self.text_encoder)

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

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature_damp = temperature_damp
        self.warmup_step = warmup_step
        self.damp_factor = damp_factor

        if self.temperature_damp:
            self.register_buffer("temp", torch.ones(1))
        else:
            self.temp = nn.Parameter(0.5*torch.ones(1))   
        
        self.negative_all_rank = negative_all_rank
        self.label_smoothing = label_smoothing
        self.max_text_tokens = 35
        
        
    def forward(self, image, caption, alpha, idx, neg_images, neg_captions, epoch=-1, epoch_threshold=-100, step=100000):
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
        # with torch.no_grad():
        #     self.temp.clamp_(0.1,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)     

        # B x D   
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    

        # negatives
        B, N, channel, width, height = neg_images.shape
        #print(f"B: {B}, N: {N}, channel: {channel}, width: {width}, height: {height}") 
        
        neg_images_embeds = self.visual_encoder(neg_images.view(B*N, channel, width, height)) 
            
        neg_images_atts = torch.ones(neg_images_embeds.size()[:-1],dtype=torch.long).to(neg_images.device)   
        #print("neg_images_atts", neg_images_atts.size())  
        # B*N x D   
        neg_images_feat = F.normalize(self.vision_proj(neg_images_embeds[:,0,:]),dim=-1)
        _, D = neg_images_feat.shape

        # B x N x D
        neg_images_feat = neg_images_feat.view(B, N, D)  
        #print("neg_images_feat", neg_images_feat.size())  


        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_text_tokens, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')       
        # B x D     
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  

        # due to no collate func, text are not correctly align
        # better to unwrap them, than adding collate func (as of now)
        # B * N
        neg_captions = [sent for sentences in neg_captions for sent in sentences]
        #print("len(neg_captions)", len(neg_captions), B*N) 

        neg_text = self.tokenizer(neg_captions, padding='max_length', truncation=True, max_length=self.max_text_tokens, 
                              return_tensors="pt").to(image.device) 
        
        neg_text_output = self.text_encoder(neg_text.input_ids, attention_mask = neg_text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        neg_text_feat = F.normalize(self.text_proj(neg_text_output.last_hidden_state[:,0,:]),dim=-1)      
        
        # B x N x D
        neg_text_feat = neg_text_feat.view(B, N, D)

        ###============== New Image-text Contrastive Learning ===================###
        
        # B x D
        pos_image_feat = image_feat
        # B x D
        pos_text_feat = text_feat

        # B x 1
        pos_i2t = (pos_image_feat * pos_text_feat).sum(-1).unsqueeze(-1)
        pos_t2i = (pos_text_feat * pos_image_feat).sum(-1).unsqueeze(-1)
        #print("pos_i2t", pos_i2t.size())
        #print("pos_t2i", pos_t2i.size())

        # B x 1 x N
        neg_i2t = pos_image_feat.unsqueeze(1) @ neg_text_feat.permute(0, 2, 1)
        neg_t2i = pos_text_feat.unsqueeze(1) @ neg_images_feat.permute(0, 2, 1)

        # B x N
        neg_i2t = neg_i2t.squeeze(1)
        neg_t2i = neg_t2i.squeeze(1)

        #print("neg_i2t", neg_i2t.size())
        #print("neg_t2i", neg_t2i.size())

        # B x N+1
        sim_i2t = torch.cat([pos_i2t, neg_i2t], dim=1) 
        sim_t2i = torch.cat([pos_t2i, neg_t2i], dim=1) 
        

        # print("sim_i2t", sim_i2t)
        # print("sim_t2i", sim_t2i)
        
        if self.temperature_damp and step > self.warmup_step:
            self.temp = self.temp * self.damp_factor
        
        sim_i2t = sim_i2t / (self.temp).clip(min=0.1, max=1)
        sim_t2i = sim_t2i / (self.temp).clip(min=0.1, max=1)

        # print("sim_i2t softmax", F.softmax(sim_i2t, dim=1))
        # print("sim_t2i softmax", F.softmax(sim_t2i, dim=1))
        # print(self.temp)
        # exit()


        with torch.no_grad():
            true_dist = torch.zeros_like(sim_i2t)
            true_dist.fill_(self.label_smoothing / (N+1))
            true_dist[:, 0] = true_dist[:, 0] + 1.0 - self.label_smoothing

            #print("true_dist", true_dist.size())
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*true_dist,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*true_dist,dim=1).mean() 
        
        # Here we can use our custom negatives (instead of finding one in batch across GPUs)
        our_negatives = False
        idx = idx.view(-1,1)
        idxs = concat_all_gather(idx)

        # skip itm in beginning
        # epoch starts with 0 (hence >=)
        if epoch >= epoch_threshold :
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
            
            if our_negatives:
                text_ids_neg = neg_text.input_ids.clone() # not sure why this was cloned above
                text_ids_neg = text_ids_neg.view(B*N, self.max_text_tokens)
                text_atts_neg = neg_text.attention_mask.view(B*N, self.max_text_tokens)
                
                # already in B*N form  
                image_embeds_neg = neg_images_embeds
                image_atts_neg = neg_images_atts
                # they pick 1 neg per img, we have N negs per image (we can reduce it too)
                # +1 is because they have added pos+neg
                neg_size = B*(2*N) 

                text_ids_all = torch.cat([encoder_input_ids.repeat(N,1), text_ids_neg],dim=0)     
                text_atts_all = torch.cat([text.attention_mask.repeat(N,1), text_atts_neg],dim=0) 

                image_embeds_all = torch.cat([image_embeds_neg,image_embeds.repeat(N,1,1)],dim=0)

                image_atts_all = torch.cat([image_atts_neg,image_atts.repeat(N,1)],dim=0)

                #print("text_id_neg", text_ids_neg.size())
                #print("text_atts_neg", text_atts_neg.size())
                #print("image_embeds_neg", image_embeds_neg.size())
                #print("image_atts_neg", image_atts_neg.size())

            else:
                # their code
                if self.negative_all_rank:    
                    # compute sample similarity
                    with torch.no_grad():                
                        mask = torch.eq(idx, idxs.t())

                        image_feat_world = concat_all_gather(image_feat)
                        text_feat_world = concat_all_gather(text_feat)

                        sim_i2t = image_feat @ text_feat_world.t() / (self.temp).clip(min=0.1, max=1)
                        sim_t2i = text_feat @ image_feat_world.t() / (self.temp).clip(min=0.1, max=1)

                        weights_i2t = F.softmax(sim_i2t,dim=1)
                        weights_i2t.masked_fill_(mask, 0)            

                        weights_t2i = F.softmax(sim_t2i,dim=1)
                        weights_t2i.masked_fill_(mask, 0)     

                    image_embeds_world = all_gather_with_grad(image_embeds) 

                    # select a negative image (from all ranks) for each text
                    image_embeds_neg = []    
                    for b in range(bs):
                        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                        image_embeds_neg.append(image_embeds_world[neg_idx])
                    image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

                    # select a negative text (from all ranks) for each image
                    input_ids_world = concat_all_gather(encoder_input_ids)
                    att_mask_world = concat_all_gather(text.attention_mask)        

                    text_ids_neg = []
                    text_atts_neg = []
                    for b in range(bs):
                        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                        text_ids_neg.append(input_ids_world[neg_idx])
                        text_atts_neg.append(att_mask_world[neg_idx])
                        
                else:
                    with torch.no_grad():                
                        mask = torch.eq(idx, idx.t())
                        
                        sim_i2t = image_feat @ text_feat.t() / (self.temp).clip(min=0.1, max=1)
                        sim_t2i = text_feat @ image_feat.t() / (self.temp).clip(min=0.1, max=1)

                        weights_i2t = F.softmax(sim_i2t,dim=1)
                        weights_i2t.masked_fill_(mask, 0)            

                        weights_t2i = F.softmax(sim_t2i,dim=1)
                        weights_t2i.masked_fill_(mask, 0)     

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

                image_atts_neg = image_atts      
                neg_size = 2*bs                 


                text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
                text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0) 

                image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)

                image_atts_all = torch.cat([image_atts_neg,image_atts],dim=0)

            #print("text_ids_all", text_ids_all.size())
            #print("text_atts_all", text_atts_all.size())
            #print("image_embeds_all", image_embeds_all.size())
            #print("image_atts_all", image_atts_all.size())

            output_neg = self.text_encoder(text_ids_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        )                         
            
            vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
            vl_output = self.itm_head(vl_embeddings)            

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(neg_size,dtype=torch.long)],
                                dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)     
        else:
            loss_itm = torch.zeros_like(loss_i2t)

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
