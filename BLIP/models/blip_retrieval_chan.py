from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from models.blip import create_vit, init_tokenizer, load_checkpoint
MASK = -10
import spacy


def run_dependency_parser(sentences):
    phrases_list = []
    nlp = spacy.load("en_core_web_sm")
    
    with torch.no_grad():

        for sentence in sentences:
            # Process the sentence with spaCy
            doc = nlp(sentence)

            # Extract noun chunks (phrases) from the dependency parse
            phrases = [chunk.text for chunk in doc.noun_chunks]
            if len(phrases)==0:
                phrases = [sentence]
            phrases_list.append(phrases)

    return phrases_list

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
        
        
        
        # Define the patches size
        self.num_max_patches = 256 # (256 patches + 1 cls)
        self.num_max_tokens = 34 # includes cls
        self.num_new_patch = 16
        # B x 256 x D -> B x 64 x D
        #self.conv_patch = nn.Conv2d(self.num_max_patches, self.num_new_patch, kernel_size=1)
        #self.conv_patch_m = nn.Conv2d(self.num_max_patches, self.num_new_patch, kernel_size=1)
        # input is Bx768X16X16 and output will be Bx768x8x8
        self.conv_patch = nn.Conv2d(768, 768, kernel_size=4, stride=4)
        self.conv_patch_m = nn.Conv2d(768, 768, kernel_size=4, stride=4)
        
        self.model_pairs = [
                    [self.vision_proj,self.vision_proj_m],
                    [self.text_proj,self.text_proj_m],
                    [self.conv_patch, self.conv_patch_m]
                    ]       
        
        self.model_pairs_frozen = [[self.visual_encoder,self.visual_encoder_m],
                                   [self.text_encoder,self.text_encoder_m]]
        
        
        self.copy_params()
        
        self.copy_params_frozen()

        # create the queue
        #self.register_buffer("image_queue", torch.randn(embed_dim, self.num_max_patches, queue_size))
        self.register_buffer("image_queue", torch.randn(embed_dim, self.num_new_patch, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.num_max_tokens, queue_size))
        self.register_buffer("len_queue", torch.full((1,queue_size),34)) # Define a queue for captions lengths
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  
        #self.register_buffer("image_queue_cls", torch.randn(embed_dim, queue_size))
        #self.register_buffer("text_queue_cls", torch.randn(embed_dim, queue_size))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        #self.image_queue_cls = nn.functional.normalize(self.image_queue_cls, dim=0)
        #self.text_queue_cls = nn.functional.normalize(self.text_queue_cls, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        #self.temp = 0.07
        
        self.negative_all_rank = negative_all_rank
        
        # Change 2: pooling strategy for patch-token level similarity
        self.pooling = LSEPooling()
        
    def forward(self, image, caption, alpha, idx):        
        # with torch.no_grad():
        #     self.temp.clamp_(0.001,0.5)
        
        # B x (1 CLS + P patches) x D
            
        image_embeds = self.visual_encoder(image) 
        B, _, _ = image_embeds.size()
        img_lens = torch.full((B,), self.num_max_patches).long().to(image.device)      

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        # B x 256 ( due to vision_proj layer)
        
        # Changed to: B x (P+1) x 256

        #image_patch_feat = F.normalize(self.vision_proj(image_embeds[:,1:,:]),dim=-1) # CLS is normalized and used as feature 
        image_emb_conv = self.conv_patch(image_embeds[:,1:,:].transpose(1,2).unflatten(2, (16,16))).flatten(2).transpose(1,2)
        image_patch_feat = F.normalize(self.vision_proj(image_emb_conv), dim=-1)
        
        '''
        Phrase representations
        '''
        
        phrases_list = run_dependency_parser(caption)
        caps_len = [len(sent) for sent in phrases_list]
        cap_lens = torch.tensor(caps_len).to(image.device).unsqueeze(0)
        phrase_all = list(chain(*phrases_list))
        phrase_tok = self.tokenizer(phrase_all, padding='max_length', truncation=True, max_length=self.num_max_tokens+1, 
                              return_tensors="pt").to(image.device)
        #cap_lens = torch.tensor([len(sent.input_ids) for sent in phrase_tok]).to(image.device).unsqueeze(0)

        phrase_out = self.text_encoder(phrase_tok.input_ids, attention_mask = phrase_tok.attention_mask,                      
                                        return_dict = True, mode = 'text')
        phrase_rep_out = phrase_out.last_hidden_state[:,0,:]
        start_index = 0
        phrase_rep = []
        for num_elements in caps_len:
            end_index = start_index + num_elements
            phrase_rep.append(phrase_rep_out[start_index:end_index])
            start_index = end_index
            
        target_seq_length = 34
        # Pad the batch to the target sequence length
        padded_batch = pad_sequence(phrase_rep, batch_first=True, padding_value=0)

        padded_batch = padded_batch.transpose(1,2)
        # If the padded sequence length is less than the target, pad the right side
        if padded_batch.size(2) < target_seq_length:
            pad_size = target_seq_length - padded_batch.size(2)
            padded_batch = F.pad(padded_batch, (0, pad_size), value=0)
        # If the padded sequence length is greater than the target, truncate the right side
        elif padded_batch.size(2) > target_seq_length:
            padded_batch = padded_batch[:, :, :target_seq_length]
        padded_batch = padded_batch.transpose(1,2) 
        
        
        '''
        CLS representation
        '''
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.num_max_tokens+1, 
                              return_tensors="pt").to(image.device) 
        #cap_lens = text.attention_mask.sum(-1).unsqueeze(0).long()
        

        # B x W x D
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')    
        # B x 256
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1) # CLS
 
        #text_token_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,1:,:]),dim=-1)
        text_token_feat = F.normalize(self.text_proj(padded_batch),dim=-1) 
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        Q_text_len = torch.cat([cap_lens, self.len_queue.clone().detach()], dim=-1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 

            # B x P x 256
            #image_patch_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,1:,:]),dim=-1) 
            image_emb_conv_m = self.conv_patch_m(image_embeds_m[:,1:,:].transpose(1,2).unflatten(2, (16,16))).flatten(2).transpose(1,2)
            image_patch_feat_m = F.normalize(self.vision_proj_m(image_emb_conv_m), dim=-1)
            
            # D(256) x P x (B + curr_queue_size)    
            image_patch_feat_m_all = torch.cat([image_patch_feat_m.permute(2,1,0),self.image_queue.clone().detach()],dim=-1)                   

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text') 
            
            # B x W x 256
            # phrases_list = run_dependency_parser(caption)
            # caps_lens = [len(sent) for sent in phrases_list]
            # phrase_all = list(chain(*phrases_list))
            # phrase_tok = self.tokenizer(phrase_all, padding='max_length', truncation=True, max_length=self.num_max_tokens+1, 
            #                     return_tensors="pt").to(image.device)
            #cap_lens = torch.tensor([len(sent.input_ids) for sent in phrase_tok]).to(image.device).unsqueeze(0)

            phrase_out_m = self.text_encoder_m(phrase_tok.input_ids, attention_mask = phrase_tok.attention_mask,                      
                                            return_dict = True, mode = 'text')
            phrase_rep_m = phrase_out_m.last_hidden_state[:,0,:]
            start_index = 0
            phrase_rep_out_m = []
            for num_elements in caps_len:
                end_index = start_index + num_elements
                phrase_rep_out_m.append(phrase_rep_m[start_index:end_index])
                start_index = end_index
            
            target_seq_length = 34
            # Pad the batch to the target sequence length
            padded_batch_m = pad_sequence(phrase_rep_out_m, batch_first=True, padding_value=0)
            padded_batch_m = padded_batch_m.transpose(1,2)
            # If the padded sequence length is less than the target, pad the right side
            if padded_batch_m.size(2) < target_seq_length:
                pad_size = target_seq_length - padded_batch_m.size(2)
                padded_batch_m = F.pad(padded_batch_m, (0, pad_size), value=0)
            # If the padded sequence length is greater than the target, truncate the right side
            elif padded_batch_m.size(2) > target_seq_length:
                padded_batch_m = padded_batch_m[:, :, :target_seq_length]
            padded_batch_m = padded_batch_m.transpose(1,2) 
            text_token_feat_m = F.normalize(self.text_proj_m(padded_batch_m),dim=-1) 

            # 256 x W x (B + curr_queue_size)
            text_token_feat_m_all = torch.cat([text_token_feat_m.permute(2,1,0), self.text_queue.clone().detach()],dim=-1)
            '''
            Uncomment for using loss from soft targets
            '''
            # B x D @ D x(B+Qt) -> B x (B+Qt)
            # changed to: (B x P x D) @ (D x W x (B+Qt)) -> B x (B+Qt)
            # B x (B+Qt)
            #sim_i2t_m = self.get_score(image_patch_feat_m, text_token_feat_m_all.permute(2,1,0), Q_text_len, is_image_first=True) / self.temp



            # B x (B+Qi)
            #sim_t2i_m = self.get_score(text_token_feat_m, image_patch_feat_m_all.permute(2,1,0), cap_lens, is_image_first=False) /self.temp



            #sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            #sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets 
            
            # image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            # image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue_cls.clone().detach()],dim=1)
            # text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            # text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue_cls.clone().detach()],dim=1)       


        # get score of each image with all texts in batch
        
        #  Bx 256 @ 256 x 57600 (comes from self.image/text_queue) need to figure out what that is

        sim_i2t = self.get_score(image_patch_feat, text_token_feat_m_all.permute(2,1,0), Q_text_len, is_image_first=True) / self.temp
        sim_t2i = self.get_score(text_token_feat, image_patch_feat_m_all.permute(2,1,0), cap_lens, is_image_first=False) / self.temp

        # sim_i2t_cls = image_feat @ text_feat_m_all / self.temp 
        # sim_t2i_cls = text_feat @ image_feat_m_all / self.temp 
         
        '''
        Uncomment for using loss from soft targets
        '''                    
        #loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        #loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
        
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
        
        # loss_i2t_cls = -torch.sum(F.log_softmax(sim_i2t_cls, dim=1)*sim_targets,dim=1).mean()
        # loss_t2i_cls = -torch.sum(F.log_softmax(sim_t2i_cls, dim=1)*sim_targets,dim=1).mean() 

        # loss_ita = (loss_i2t+loss_t2i)/2
        
        # loss_i2t = (1-alpha)*loss_i2t + alpha*loss_i2t_cls
        # loss_t2i = (1-alpha)*loss_t2i + alpha*loss_t2i_cls
        
        idxs = concat_all_gather(idx)
        lenses = concat_all_gather(cap_lens)
        # self._dequeue_and_enqueue(image_patch_feat_m, text_token_feat_m, idxs, lenses, image_feat_m, text_feat_m)   
        self._dequeue_and_enqueue(image_patch_feat_m, text_token_feat_m, idxs, lenses)        

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

        
        if self.negative_all_rank:    
            # compute sample similarity
            with torch.no_grad():                
                mask = torch.eq(idx, idxs.t())
                '''

                image_patch_feat_world = concat_all_gather(image_patch_feat)
                text_token_feat_world = concat_all_gather(text_token_feat)

                sim_i2t = image_patch_feat @ text_token_feat_world.t() / self.temp 
                sim_t2i = text_token_feat @ image_patch_feat_world.t() / self.temp 
                '''
                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp 
                sim_t2i = text_feat @ image_feat_world.t() / self.temp 
                
                

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            image_embeds_world = all_gather_with_grad(image_embeds) 

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []    
            for b in range(bs):
                normalized_probs = torch.softmax(weights_t2i[b], dim=0)
                neg_idx = torch.multinomial(normalized_probs, 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from all ranks) for each image
            input_ids_world = encoder_input_ids
            att_mask_world = text.attention_mask 

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                normalized_probs = torch.softmax(weights_t2i[b], dim=0)
                neg_idx = torch.multinomial(normalized_probs, 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                
        else:
            with torch.no_grad():                
                mask = torch.eq(idx, idx.t())
                '''
                # sim_i2t = image_patch_feat @ text_token_feat.t() / self.temp 
                sim_i2t = self.get_score(image_patch_feat, text_token_feat, cap_lens, is_image_first=True)
                
                # sim_t2i = text_token_feat @ image_patch_feat.t() / self.temp 
                sim_t2i = self.get_score(text_token_feat, image_patch_feat, cap_lens, is_image_first=False)
                #print("sim_t2i", sim_t2i)
                '''
                sim_i2t = image_feat @ text_feat.t() / self.temp 
                sim_t2i = text_feat @ image_feat.t() / self.temp 

                weights_i2t = F.softmax(sim_i2t,dim=1)
                weights_i2t.masked_fill_(mask, 0)            

                weights_t2i = F.softmax(sim_t2i,dim=1)
                weights_t2i.masked_fill_(mask, 0)     

            # select a negative image (from same rank) for each text
            image_embeds_neg = []    
            for b in range(bs):
                normalized_probs = torch.softmax(weights_t2i[b], dim=0)
                #print("weights_t2i", weights_t2i)
                #print("normalized_probs", normalized_probs)
                neg_idx = torch.multinomial(normalized_probs, 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                normalized_probs = torch.softmax(weights_t2i[b], dim=0)
                neg_idx = torch.multinomial(normalized_probs, 1).item()
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

        with torch.no_grad():
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
                
    def copy_params_frozen(self):
        for model_pair in self.model_pairs_frozen:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient   
                param.requires_grad = False 

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_patch_feat, text_token_feat, idxs, cap_lens, image_feat=None, text_feat=None):
        # gather keys before updating queue
        image_patch_feats = concat_all_gather(image_patch_feat)
        text_token_feats = concat_all_gather(text_token_feat)
        if image_feat is not None:
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        

        batch_size = image_patch_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, :, ptr:ptr + batch_size] = image_patch_feats.permute(2,1,0)
        self.text_queue[:, :, ptr:ptr + batch_size] = text_token_feats.permute(2,1,0)
        
        if image_feat is not None:
            self.image_queue_cls[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue_cls[:, ptr:ptr + batch_size] = text_feats.T
        #print("idx", cap_lens.shape, idxs.size())
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.t()
        self.len_queue[:, ptr:ptr + batch_size] = cap_lens.view(1,-1)
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr 

        
    # calculate the fine-grained similarity according to the given images and captions
    def get_score(self, imgs, caps, cap_lens, is_image_first):
        """similarity score

        Args:
            imgs (_type_): x modality feats
            caps (_type_): y modality feats 
            cap_lens (_type_): num text token 1 x B
            is_image_first (bool): True if imgs is image modality else False

        Returns:
            torch.Tensor: patch-token similarity score
        """
        Bi, P, D = imgs.shape
        Bc, W, D = caps.shape
        
        imgs = imgs.reshape(Bi*P, D)
        caps = caps.reshape(Bc*W, D).t()
        sims = torch.matmul(imgs,caps)
        # text_queue D x L x Qt
        # text_queue mask Qt
        # Bi x Qt x P x W
        sims = sims.reshape(Bi, P, Bc, W).permute(0,2,1,3)
        
        mask = self.get_fgmask(Bi if is_image_first else Bc, imgs.device, cap_lens, is_image_first)

        #print("mask", mask.size())
        #print("sims", sims.size())
        sims = sims.masked_fill(mask == 0, MASK)
        if is_image_first:
            ddim=-2
        else:
            ddim=-1
        #ddim=-1
        #sims = F.softmax(sims, dim=ddim)
        
        # Bi x Bc x P
        #print("max", sims.max(dim=-1))
        sims = sims.max(dim=-1)[0]
        # print("before pooling")
        # print(sims)
        #print("tmp sims", sims)
        

        # Bi x Bc == B x (B+Qt)
        sims = self.pooling(sims)
        

        # print("after pooling")
        # print(sims)
        # exit()

        return sims


    # calculate the mask of fine-grained similarity according to the given images length and captions length
    def get_fgmask(self, bi, img_device, cap_lens, is_image_first):
        
        # bi = img_lens.shape[0]
        cap_lens = cap_lens.squeeze()
        bc = cap_lens.shape[0]
        #max_r = self.num_max_patches # int(img_lens.max())
        max_r = self.num_new_patch
        max_w = self.num_max_tokens # int(cap_lens.max())

        # mask_i = torch.arange(max_r).expand(bi, max_r).to(img_lens.device)
        # mask_i = (mask_i<img_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(img_lens.device)
        # mask_i = mask_i.reshape(bi*max_r,1)
        mask_i = torch.ones((bi*max_r, 1)).to(img_device)


        mask_c = torch.arange(max_w).expand(bc,max_w).to(cap_lens.device)
        mask_c = (mask_c<cap_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(cap_lens.device)
        mask_c = mask_c.reshape(bc*max_w,1)

        if is_image_first:
            mask = torch.matmul(mask_i,mask_c.t()).reshape(bi, max_r, bc, max_w).permute(0,2,1,3)
        else:
            mask = torch.matmul(mask_c, mask_i.t()).reshape(bc, max_w, bi, max_r).permute(0,2,1,3)
        return mask
    

# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, temperature=0.1):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        sims = sims/temperature # this fixes nan error investigate
        sims = torch.logsumexp(sims,dim=-1)*temperature
        #print("after LSE", sims)
        # if MASK is not None:
        #     sims[sims == MASK] = 0  # Set masked values to 0 or another appropriate value
        # sims = sims.mean(dim=-1)  # Calculate the mean along the last dimension
        # return sims
        # lens = (sims!=MASK).sum(dim=-1)
        # sims[sims==MASK] = 0
        # sims = sims.sum(dim=-1)/lens
        return sims

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
