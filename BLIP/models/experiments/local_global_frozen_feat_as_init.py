from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from models.experiments.alignment_module import multi_alignment_block, residual_alignment_block

from models.blip import create_vit, init_tokenizer, load_checkpoint

import os
'''
TODO: 1. single alignment module
2. 3 alignment module - shared alignment module
'''

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
                 config=None
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
        

        # print config
        for k,v in config.items():
            print(f"{k}={v}")
        
        # codebooks
        self.entries = config['codes']
        self.shared_codebooks = config['use_shared_codebooks']
        self.shared_attention = config['use_shared_attention']
        
        self.config = config

        # in multiple layers, all layers attend to previous layer codebook
        # and frozen features 
        # (as opposed to all layers attending to codebooks and previous frozen features)
        self.update_x_cross = config['update_x_cross']
        self.swap_attn_inps = config.get('swap_attn_inps', False)
        self.skip_x = config.get('skip_x', False)
        self.retain_org_x = config.get('retain_org_x', False)

        if self.shared_codebooks:
            self.v_codebook = nn.Embedding(self.entries, embed_dim)
            self.l_codebook = self.v_codebook

            self.v_codebook_m = nn.Embedding(self.entries, embed_dim)
            self.l_codebook_m = self.v_codebook_m

        else:
            self.v_codebook = nn.Embedding(self.entries, embed_dim)
            self.l_codebook = nn.Embedding(self.entries, embed_dim)

            self.v_codebook_m = nn.Embedding(self.entries, embed_dim)
            self.l_codebook_m = nn.Embedding(self.entries, embed_dim)
            
        self.num_attn_blocks = config['num_alignment_modules']
        
        if config.get('use_residual',False):
            alignment_mod = residual_alignment_block
        else:
            alignment_mod = multi_alignment_block
            
        if self.shared_attention:
            self.codebook_attention_v = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )
            self.codebook_attention_l = self.codebook_attention_v
            
            self.codebook_attention_v_m = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )
            self.codebook_attention_l_m = self.codebook_attention_v_m

        else:
            self.codebook_attention_v = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )
            self.codebook_attention_l = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )

            self.codebook_attention_v_m = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )
            self.codebook_attention_l_m = alignment_mod(
                embed_dim, 
                self.num_attn_blocks, 
                update_x_cross=self.update_x_cross, 
                swap_attn_inps=self.swap_attn_inps,
                skip_x=self.skip_x,
                retain_org_x=self.retain_org_x,
            )

          
        # freeze the pretrained encoders and initialize the new encoders    
        self.model_pairs_pretrained = [[self.visual_encoder_new, self.visual_encoder],
                                       [self.vision_proj_new, self.vision_proj],
                                       [self.text_encoder_new, self.text_encoder],
                                       [self.text_proj_new, self.text_proj]]
        # self.initialize()
        
        self.freeze_encoders = config['freeze_encoders']
        
        self.freeze_models = []
        
        if config['concat_text_local_to_baseline']:
            self.concat_proj = nn.Linear(embed_dim*2, embed_dim)
            self.model_pairs_concat_baseline = [[self.visual_encoder_new, self.visual_encoder],
                                       [self.vision_proj_new, self.vision_proj],
                                       [self.text_encoder_new, self.text_encoder],
                                       [self.text_proj_new, self.text_proj]]
            self.unfreeze_models = [self.visual_encoder,
                                    self.vision_proj,
                                    self.text_encoder,
                                    self.text_proj,
                                    self.concat_proj]
        
        if self.freeze_encoders:
            self.freeze_models += [self.visual_encoder_new,
                                self.vision_proj_new,
                                self.text_encoder_new,
                                self.text_proj_new,
                                ]
            
            if config["freeze_codebooks"]:
                self.freeze_models += [self.v_codebook, self.l_codebook]

            self.freeze()
        
        # freeze the paramters of momentum encoder
        self.model_pairs = [[self.visual_encoder_new,self.visual_encoder_m],
                            [self.vision_proj_new,self.vision_proj_m],
                            [self.text_encoder_new,self.text_encoder_m],
                            [self.text_proj_new,self.text_proj_m],
                            [self.v_codebook,self.v_codebook_m],
                            [self.l_codebook,self.l_codebook_m],
                            [self.codebook_attention_v, self.codebook_attention_v_m],
                            [self.codebook_attention_l, self.codebook_attention_l_m],
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
        self.fine_temp = nn.Parameter(0.5*torch.ones([]))
        self.local_temp = nn.Parameter(0.5*torch.ones([]))
        self.similarity_threshold = 0.1

        self.use_codebooks = config['use_codebooks']
        self.label_smoothing = config['use_label_smoothing']
        self.smoothing_mass = config.get("smoothing_mass", 0.1)

        self.use_attn_mask = config['use_attn_mask']
        self.layer_no = config['image_feature_layer_num']
        print(f"Using layer for images: {self.layer_no}")
        # self.layer_no = 'patch_and_last'

        
    def forward(self, image, caption, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            self.fine_temp.clamp_(0.0001,0.1)
            self.local_temp.clamp_(0.0001,0.1)
            

        v_cls_frozen, l_cls_frozen = self.get_frozen_feat(image, caption)
        v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask = self.get_trainable_feat(image, caption, momentum=False)
        
        ###============== Image-text Contrastive Learning ===================###
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
            
        # ---------- Local similarity matrix -------- #  
        if not self.swap_attn_inps or self.config.get('concat_text_local_to_baseline', False):
  
            # similarity calculation BxTxP

            similarity = torch.einsum('btd,bpd->btp',l_word, v_patch)
            
            # min-max normalisation 

            # with swapped attn inps not need for this
            eps = 1e-6
            similarity = (similarity-torch.min(similarity, dim=-1).values.unsqueeze(-1))/(torch.max(
                similarity, dim=-1).values.unsqueeze(-1) - torch.min(similarity, dim=-1).values.unsqueeze(-1) + eps)
            
            # thresholding
            similarity = torch.where(similarity < self.similarity_threshold, 0.0, similarity)

            
            # alignment-weighting
            v_align_weights = similarity/(torch.sum(similarity, dim=-1).unsqueeze(-1) + eps)
            v_patch = torch.einsum('btp,bpd->btd', v_align_weights, v_patch)


        # local loss
        local_mask = language_mask
        if self.swap_attn_inps:
            local_mask = None
        
        # if self.config.get('concat_text_local_to_baseline', False):
        #     local_lv = 0
        #     local_vl = 0
        # else:
        local_vl = self.local_pairwise_contrastive_loss(v_patch, l_word, local_mask)
        local_lv = self.local_pairwise_contrastive_loss(l_word, v_patch, local_mask)
        
        # ----------- Global Similarity matrices ----------- #
        cls_sim_i2t = v_cls @ l_cls_m_all / self.temp 
        cls_sim_t2i = l_cls @ v_cls_m_all / self.temp
        
        if self.config.get('concat_text_local_to_baseline', False):
            v_patch_cls = v_cls_frozen
            l_word_cls = F.normalize(self.concat_proj(torch.cat([l_word_cls, l_cls_frozen], dim=-1)), dim=-1)
        
        pooled_sim_i2t = v_patch_cls @ l_word_cls.t()
        pooled_sim_t2i = l_word_cls @ v_patch_cls.t()

        torch.set_printoptions(precision=3)

        
        # print("v_patch_cls")
        # print(v_patch_cls)
        # print("l_word_cls_m_all")
        # print(l_word_cls_m_all)
        print("pooled_sim_i2t")
        print(pooled_sim_i2t)
        # print("pooled_sim_i2t softmax")
        # print(pooled_sim_i2t.softmax(-1))
        # print("std")
        # print(pooled_sim_i2t.std(-1))
        # print("max")
        # print(pooled_sim_i2t.max(-1).values)
        pooled_sim_i2t = pooled_sim_i2t / self.fine_temp
        pooled_sim_t2i = pooled_sim_t2i / self.fine_temp

        print("pooled_sim_i2t after tmp")
        print(pooled_sim_i2t)
        
        print("pooled_sim_i2t after tmp softmax")
        torch.set_printoptions(precision=3)
        print(pooled_sim_i2t.softmax(-1))

        # print("caption", caption)
        
        # print("*"*100)

        with torch.no_grad():
            B, _ = sim_targets.size()

            sim_targets2 = torch.eye(B).to(sim_targets.device)

            if self.label_smoothing:
                # remove smoothing mass from 1's
                sim_targets2 = sim_targets2*(1-self.smoothing_mass)
                # uniformly redistribute smoothing mass to all
                sim_targets2 = sim_targets2 + self.smoothing_mass/B


        # ----------- Losses -------------- #
        cls_loss_i2t = -torch.sum(F.log_softmax(cls_sim_i2t, dim=1)*sim_targets,dim=1).mean()
        cls_loss_t2i = -torch.sum(F.log_softmax(cls_sim_t2i, dim=1)*sim_targets,dim=1).mean()
        
        pooled_loss_i2t = -torch.sum(F.log_softmax(pooled_sim_i2t, dim=1)*sim_targets2, dim=-1).mean()
        pooled_loss_t2i = -torch.sum(F.log_softmax(pooled_sim_t2i, dim=1)*sim_targets2, dim=-1).mean()
        
        distance_v = (1-torch.sum(v_cls_frozen*v_cls, dim=-1)).mean()
        distance_l = (1-torch.sum(l_cls_frozen*l_cls, dim=-1)).mean()
        
        l1 = 0.5*(cls_loss_i2t + cls_loss_t2i)
        l2_global = 0.5*(pooled_loss_i2t + pooled_loss_t2i)
        l2_local = 0.5*(local_vl + local_lv)
        l2 = 0.5*(l2_local + l2_global)
        l3 = 0.5*(distance_l + distance_v)
        
        # loss = (l1 + l2 + l3)/3.0
        loss = l2_global
        # loss = 0.7*l2_global + 0.3
    
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
            "L2_global":l2_global,
            "fine_temp": self.fine_temp,
            "local_temp": self.local_temp,
        }     

        return loss_dict
    
    def get_frozen_feat(self, i, t):
        # TODO: Do we need to return language mask as well 
        # image cls
        image_last_frozen, image_patch_frozen = self.visual_encoder(i, return_layer = self.layer_no)
        v_cls_frozen = F.normalize(self.vision_proj(image_last_frozen[:,0,:]),dim=-1)
        
        # text cls
        text = self.tokenizer(t, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(i.device)
        text_output_frozen, word_embed_frozen = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
        l_cls_frozen = F.normalize(self.text_proj(text_output_frozen.last_hidden_state[:,0,:]),dim=-1)
        
        return v_cls_frozen, l_cls_frozen
                
    
    def get_trainable_feat(self, i, t, momentum=False, save_wts=False, model_name=""):
        code_tokens = torch.arange(0, self.entries).unsqueeze(0).expand(i.shape[0], -1).to(i.device)
        text = self.tokenizer(t, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(i.device)
        if not momentum:
            # patch embeddings from frozen image encoder
            """
            Frozen features
            """
            image_last, image_patch = self.visual_encoder_new(i, return_layer = self.layer_no)
            # _, image_patch = self.visual_encoder(i, return_layer = self.layer_no)
            v_patch = self.v_patch_proj(image_patch)

            if self.layer_no != "patch_and_last" and self.layer_no > -1:
                # remove cls
                v_patch = v_patch[:, 1:, :]

            # pool the image patches
            B,T,D = v_patch.shape
            v_patch = v_patch.view(B,16,16,D).permute(0,3,1,2)
            v_patch = self.pool_patch(v_patch).permute(0,2,3,1).view(B,16,D)
        
            # word embeddings from frozen text encoder
            text_output, word_embed = self.text_encoder_new(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            # _, word_embed = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
            #                                 return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            language_mask = text.attention_mask[:,1:]
            l_word = self.l_word_proj(word_embed[:,1:,:])
        
            # cls from trainable encoders
            v_cls = F.normalize(self.vision_proj_new(image_last[:,0,:]),dim=-1)
            l_cls = F.normalize(self.text_proj_new(text_output.last_hidden_state[:,0,:]),dim=-1)
            
            # pooled cls after attention over codebooks for new patch and word embeddings
            if self.use_codebooks:
                v_codes = self.v_codebook(code_tokens)
                l_codes = self.l_codebook(code_tokens)
            else:
                v_codes = v_patch     
                l_codes = l_word


            """
            Save attention weights
            """

            save_path = None
            if save_wts:
                token_name = self.tokenizer.batch_decode([[x] for x in text.input_ids[0] if x!=0])
                save_path = "__".join(token_name)
                save_path = "IMAGE_"+save_path
                save_dir = f"experiments/visualize/{model_name}"
                save_path = os.path.join(save_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)
                
            if self.use_codebooks:
                v_patch = self.codebook_attention_v(v_patch, v_codes, save_path=save_path)

            if self.use_attn_mask:
                if self.swap_attn_inps:
                    # B x T -> B x 100 x T
                    attn_mask = (~(language_mask.bool())).unsqueeze(1).expand(-1, self.entries, -1)
                else:
                    # B x T x T
                    # one represents padded
                    attn_mask = ~(language_mask.bool().unsqueeze(-1) * language_mask.bool().unsqueeze(1))
            else:
                attn_mask = None

            """
            Save attention weights
            """

            if save_wts:
                token_name = self.tokenizer.batch_decode([[x] for x in text.input_ids[0] if x!=0])
                save_path = "__".join(token_name)
                save_path = "TEXT_"+save_path
                save_dir = f"experiments/visualize/{model_name}"
                save_path = os.path.join(save_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)

            if self.use_codebooks:
                l_word = self.codebook_attention_l(l_word, l_codes, mask = attn_mask, save_path=save_path)

            v_patch_cls = F.normalize(torch.mean(v_patch, dim=1), dim=-1)
            # v_patch_cls = torch.mean(v_patch, dim=1)

            if self.swap_attn_inps:
                # This gives codebook sized repr -derived from token-embeds
                # (previously it was token-sized repr derived from codebooks)
                # B x 100 x D
                # no masking needed
                l_word_cls = l_word.mean(dim=1)
                l_word_cls = F.normalize(l_word_cls, dim=-1)
            else:
                l_word = l_word.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
                l = language_mask.sum(-1).unsqueeze(-1)

                assert (l==0).sum() == 0, "divide by zero"

                l_word_cls = F.normalize(torch.sum(l_word, dim=1)/l, dim=-1)
                # l_word_cls = torch.sum(l_word, dim=1)/l
            
            return v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask
        
        else:
            # patch embeddings from trainable image encoder
            image_last_m, image_patch_m = self.visual_encoder_m(i, return_layer = self.layer_no)
            v_patch_m = self.v_patch_proj_m(image_patch_m)
    
            if self.layer_no != "patch_and_last" and self.layer_no > -1:
                # remove cls
                v_patch_m = v_patch_m[:, 1:, :]

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
            if self.use_codebooks:
                v_codes_m = self.v_codebook_m(code_tokens)
                l_codes_m = self.l_codebook_m(code_tokens)
            else:
                v_codes_m = v_patch_m
                l_codes_m = l_word_m
                
            if self.use_codebooks:
                v_patch_m = self.codebook_attention_v_m(v_patch_m, v_codes_m)

            if self.use_attn_mask:
                if self.swap_attn_inps:
                    # B x T -> B x 100 x T
                    attn_mask = (~(language_mask.bool())).unsqueeze(1).expand(-1, self.entries, -1)
                else:
                    # B x T x T
                    # one represents padded
                    attn_mask = ~(language_mask.bool().unsqueeze(-1) * language_mask.bool().unsqueeze(1))
            else:
                attn_mask = None
                
            if self.use_codebooks:
                l_word_m = self.codebook_attention_l_m(l_word_m, l_codes_m, mask=attn_mask)
                
            v_patch_cls_m = F.normalize(torch.mean(v_patch_m, dim=1), dim=-1)
            if self.swap_attn_inps:
                # This gives codebook sized repr -derived from token-embeds
                # (previously it was token-sized repr derived from codebooks)
                # B x 100 x D
                # no masking needed
                l_word_m = l_word_m.mean(dim=1)
                l_word_cls_m = F.normalize(l_word_m, dim=-1)
            else:
                l_word_m = l_word_m.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
                l = language_mask.sum(-1).unsqueeze(-1)

                assert (l==0).sum() == 0, "divide by zero"

                l_word_cls_m = F.normalize(torch.sum(l_word_m, dim=1)/l, dim=-1)

            return v_patch_cls_m, l_word_cls_m, v_cls_m, l_cls_m, language_mask
            
        
    def get_untrained_feats(self, i, t, momentum=False, save_wts=False, model_name=""):
        code_tokens = torch.arange(0, self.entries).unsqueeze(0).expand(i.shape[0], -1).to(i.device)
        text = self.tokenizer(t, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt").to(i.device)
        if not momentum:
            # patch embeddings from frozen image encoder
            """
            Frozen features
            """
            image_last, image_patch = self.visual_encoder(i, return_layer = self.layer_no)
            # _, image_patch = self.visual_encoder(i, return_layer = self.layer_no)
            # v_patch = self.v_patch_proj(image_patch)
            v_patch = image_patch

            if self.layer_no != "patch_and_last" and self.layer_no > -1:
                # remove cls
                v_patch = v_patch[:, 1:, :]

            # pool the image patches
            B,T,D = v_patch.shape
            # v_patch = v_patch.view(B,16,16,D).permute(0,3,1,2)
            # v_patch = self.pool_patch(v_patch).permute(0,2,3,1).view(B,16,D)
        
            # word embeddings from frozen text encoder
            text_output, word_embed = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            # _, word_embed = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
            #                                 return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
            language_mask = text.attention_mask[:,1:]
            # l_word = self.l_word_proj(word_embed[:,1:,:])
            l_word = word_embed[:,1:,:]
        
            # cls from trainable encoders
            v_cls = F.normalize(self.vision_proj_new(image_last[:,0,:]),dim=-1)
            l_cls = F.normalize(self.text_proj_new(text_output.last_hidden_state[:,0,:]),dim=-1)
            

            v_patch_cls = F.normalize(torch.mean(v_patch, dim=1), dim=-1)

            l_word = l_word.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
            l = language_mask.sum(-1).unsqueeze(-1)

            assert (l==0).sum() == 0, "divide by zero"

            l_word_cls = F.normalize(torch.sum(l_word, dim=1)/l, dim=-1)
            
            return v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask
    
    def local_pairwise_contrastive_loss(self, a, b, mask):
        # mask (B x T): 1 implies unmasked
        batch_size, seq_len, _ = a.shape

        with torch.no_grad():
            
            labels = torch.eye(seq_len).to(a.device).unsqueeze(0).expand(batch_size, -1, -1)

            if self.label_smoothing:
                labels = labels*(1-self.smoothing_mass)
                labels = labels + (self.smoothing_mass/seq_len)

                # Assertion fails, I think precision issue
                # manually looked at labels; looks fine.
                tolerance = 1e-6
                assert torch.all((labels.sum(-1) - 1.0) < tolerance), "prob mass doesn't sum to 1"

        # (B x C x D) @ (B x C x D) -> B x C x C
        logits = torch.einsum('bmd,bnd->bmn',a,b) / self.local_temp

        if mask is not None:
            # B x T x T
            mask_logits = ~(mask.bool().unsqueeze(-1) * mask.bool().unsqueeze(1))
        
        neg_inf = torch.finfo(logits.dtype).min

        if mask is not None:
            logits = logits.masked_fill(mask_logits, neg_inf)
        
        # loss
        log_prob = F.log_softmax(logits, dim=-1)

        if mask is not None:
            log_prob = log_prob.masked_fill(mask_logits, 0.)

        loss = -log_prob * labels

        # B x C or B x T
        loss = loss.sum(-1)

        if mask is not None:
            loss = loss.sum() / mask.sum()        
            assert mask.sum() > 0, "everything is masked"
        else:
            loss = loss.mean()

        return loss
    
    @torch.no_grad()
    def inference(self, image, caption, save_wts=False, model_name=""):
        # image embedding
        v_cls_frozen, l_cls_frozen = self.get_frozen_feat(image, caption)
        v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask = self.get_trainable_feat(
                                    image, caption, momentum=False,
                                    save_wts=save_wts,
                                    model_name=model_name,
                                )
        if self.config.get('concat_text_local_to_baseline', False):
            v_patch_cls = v_cls_frozen
            l_word_cls = F.normalize(self.concat_proj(torch.cat([l_word_cls, l_cls_frozen], dim=-1)), dim=-1)
        
        # v_patch, l_word, v_patch_cls, l_word_cls, v_cls, l_cls, language_mask = self.get_untrained_feats(
        #                             image, caption, momentum=False,
        #                             save_wts=save_wts,
        #                         )

        # image_patch, image_last = self.visual_encoder_new(image, return_layer = self.layer_no)
        # v_cls = F.normalize(self.vision_proj_new(image_last[:,0,:]),dim=-1)
        # v_patch = self.v_patch_proj(image_patch)

        # if self.layer_no != "patch_and_last" and self.layer_no > -1:
        #     # remove cls
        #     v_patch = v_patch[:, 1:, :]

        # # pool the image patches
        # B,T,D = v_patch.shape
        # v_patch = v_patch.view(B,16,16,D).permute(0,3,1,2)
        # v_patch = self.pool_patch(v_patch).permute(0,2,3,1).view(B,16,D)

        # # text embedding
        # text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
        #                       return_tensors="pt").to(image.device) 
        # text_output, word_embed = self.text_encoder_new(text.input_ids, attention_mask = text.attention_mask,                      
        #                                 return_dict = True, mode = 'text', return_word_embeddings_and_last_state=True)
        # l_cls = F.normalize(self.text_proj_new(text_output.last_hidden_state[:,0,:]),dim=-1)
        # language_mask = text.attention_mask[:,1:]
        # l_word = self.l_word_proj(word_embed[:,1:,:])

        # # codebook embeddng
        # code_tokens = torch.arange(0, self.entries).unsqueeze(0).expand(l_cls.shape[0], -1).to(image.device)
        # if self.use_codebooks:
        #     v_codes = self.v_codebook(code_tokens)
        #     l_codes = self.l_codebook(code_tokens)
        # else:
        #     v_codes = v_patch
        #     l_codes = l_word

        # v_patch_cls = F.normalize(torch.mean(self.codebook_attention_v(v_patch, v_codes, None), dim=1), dim=-1)

        # l_word = l_word.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
        # l = language_mask.sum(-1).unsqueeze(-1)
        # l_word = self.codebook_attention_l(l_word, l_codes, None)
        # l_word_cls = F.normalize(torch.sum(l_word, dim=1)/l, dim=-1)

        
        return v_cls_frozen, l_cls_frozen, v_patch_cls, l_word_cls
    
    @torch.no_grad()
    def freeze(self):
        for model in self.freeze_models:
            for param in model.parameters():
                param.require_grad = False

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
    def initialize_concat_baseline(self):
        for model_pair in self.model_pairs_concat_baseline:
            for param_pre, param in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param.data.copy_(param_pre.data) # initialize
                param_pre.requires_grad = False # not update by gradient
                param.requires_grad = True
                
        for param in self.parameters():
            param.requires_grad = False
            
        for model in self.unfreeze_models:
            for param in model.parameters():
                param.requires_grad=True

            
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


def blip_retrieval(pretrained='',inference=False,**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    # print('yeahh')
    # print(model.visual_encoder.blocks[0].norm1.weight)
    # print(model.visual_encoder_new.blocks[0].norm1.weight)

    #assert torch.all(model.visual_encoder.blocks[0].norm1.weight == model.visual_encoder_new.blocks[0].norm1.weight), "some weights not same"

    if not inference:
        if kwargs['config'].get('concat_text_local_to_baseline',False):
            model.initialize_concat_baseline()
        else:
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
