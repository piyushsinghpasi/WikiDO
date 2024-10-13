from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from models.sparse_attention import SparseAttention
from models.blip import create_vit, init_tokenizer, load_checkpoint
from models.attention import MultiHeadedAttention

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
        self.local_temp = nn.Parameter(0.07*torch.ones([]))
        self.global_temp = nn.Parameter(0.07*torch.ones([]))  
        self.similarity_threshold = 0.7 
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.v_proj = nn.Linear(vision_width, embed_dim)
        self.l_proj = nn.Linear(text_width, embed_dim)

        self.model_pairs = [self.visual_encoder, self.text_encoder, self.vision_proj, self.text_proj]
        self.freeze()
        self.codebook_entries = 200
        self.codebook_dim = embed_dim
        self.codebook_l = nn.Embedding(self.codebook_entries, self.codebook_dim)

        self.codebook_attention_l = MultiHeadedAttention(1, embed_dim, 0.2, ['query', 'key'])
        self.codebook_attention_v = MultiHeadedAttention(1, embed_dim, 0.2, ['query', 'key'])


    def forward(self, image, caption, idx=None, inference=False):
        with torch.no_grad():
            self.local_temp.clamp_(0.001,0.5)
            self.global_temp.clamp_(0.001,0.5)
            
        # get image cls
        v_cls = self.visual_encoder(image)
        v_cls = F.normalize(self.vision_proj(v_cls[:,0,:]),dim=-1)  
        
        # get patch embeddings BxPxD
        v_patch_embed = self.visual_encoder.patch_embed(image) 
        
        # get word embeddings BxTxD
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device)
        language_mask = text.attention_mask
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text', return_word_embeddings=True, 
                                        return_only_word_emb=True)
        l_token_embed = text_output
        
        # get text cls
        text_output_cls = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')
        l_cls = F.normalize(self.text_proj(text_output_cls.last_hidden_state[:,0,:]),dim=-1)    
        
        
        l_token_embed = self.l_proj(l_token_embed)
        v_patch_embed = self.v_proj(v_patch_embed)

        # attention over codebooks
        batch_size, seq_len, _ = l_token_embed.shape
        codebook_embeddings_l = self.codebook_l.weight.unsqueeze(0).expand(batch_size,-1,-1)
        codebook_embeddings_v = self.codebook_l.weight.unsqueeze(0).expand(batch_size,-1,-1)
        print("="*100)
        print("attentions")
        l_token_embed, attn_l = self.codebook_attention_l(l_token_embed, codebook_embeddings_l, codebook_embeddings_l, None)
        v_patch_embed, attn_v = self.codebook_attention_v(v_patch_embed, codebook_embeddings_v, codebook_embeddings_v, None)

        # similarity calculation BxTxP
        similarity = torch.einsum('btd,bpd->btp',l_token_embed, v_patch_embed)
        
        # min-max normalisation 
        similarity = (similarity-torch.min(similarity, dim=-1).values.unsqueeze(-1))/(torch.max(
            similarity, dim=-1).values.unsqueeze(-1) - torch.min(similarity, dim=-1).values.unsqueeze(-1))
        
        # thresholding
        similarity = torch.where(similarity < self.similarity_threshold, 0.0, similarity)
        
        # alignment-weighting
        v_align_weights = similarity/torch.sum(similarity, dim=-1).unsqueeze(-1)
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)
        
        # pooling for global rep
        v_pooled = l_grouped_v_patch_embed.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
        l_pooled = l_token_embed.masked_fill(~language_mask.bool().unsqueeze(-1), 0)
        l = language_mask.sum(-1).unsqueeze(-1)
        
        # v_pooled = F.normalize(torch.sum(v_pooled, dim=1)/l, dim=-1)
        # l_pooled = F.normalize(torch.sum(l_pooled, dim=1)/l, dim=-1)
        
        v_pooled = F.normalize(torch.sum(v_patch_embed, dim=1)/l, dim=-1)
        l_pooled = F.normalize(torch.sum(l_pooled, dim=1)/l, dim=-1)


        if inference:
            return v_cls, l_cls, v_pooled, l_pooled
        
        # global loss claculation
        print("="*100)
        print("VL")
        global_vl = self.global_pairwise_contrastive_loss(v_pooled, l_pooled)
        print("="*100)
        print("LV")
        global_lv = self.global_pairwise_contrastive_loss(l_pooled, v_pooled)
        
        # cosine loss with cls 
        cosine_v = self.cosine_loss(v_pooled, v_cls)
        cosine_l = self.cosine_loss(l_pooled, l_cls)
        
        # L2 normalisation
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, dim=-1)
        l_token_embed = F.normalize(l_token_embed, dim=-1)
        
        # local loss calculation
        local_vl = self.local_pairwise_contrastive_loss(l_grouped_v_patch_embed, l_token_embed, language_mask)
        local_lv = self.local_pairwise_contrastive_loss(l_token_embed, l_grouped_v_patch_embed, language_mask) 


        entropy_l = self.entropy(attn_l, mask=language_mask)
        entropy_v = self.entropy(attn_v)

        ukl_l = self.KL_with_uniform_loss(attn_l, mask=language_mask)
        ukl_v = self.KL_with_uniform_loss(attn_v)
        
        loss_dict  = {
            "local_lv":local_lv,
            "local_vl":local_vl,
            "global_lv":global_lv,
            "global_vl":global_vl,
            "cosine_v":cosine_v,
            "cosine_l":cosine_l,
            "entropy_l": entropy_l,
            "entropy_v": entropy_v,
            "ukl_l": ukl_l,
            "ukl_v": ukl_v,
        } 

        return loss_dict
    

    def entropy(self, a, mask=None):
        # 64, 1, 35, 100
        # bactch, head, token, entries
        eps = 1e-5
        a = a + eps
        ent = -a * a.log()

        # 64, 1, 35
        ent = ent.sum(-1)

        l = None

        if mask is not None:
            # mask -> B x 35 (Max)
            ent = ent.squeeze(1)
            ent = ent.masked_fill(~mask.bool(), 0.)
            l = mask.sum()

        if l is not None:
            return ent.sum() / l
    
        return ent.mean()

    def KL_with_uniform_loss(self, inp, mask=None):
        # batch, token, codebook entries
        inp = inp.squeeze(1)

        # kl loss expects inp in log space
        log_prob = inp.log()

        # target distribution
        with torch.no_grad():
            u = torch.ones_like(log_prob) / self.codebook_entries    

        loss_kl_fct = nn.KLDivLoss(reduction="none")
        loss_kl = loss_kl_fct(log_prob, u)


        # B x Token x Attn -> B x Token
        loss_kl = loss_kl.sum(-1)
        
        l = None

        if mask is not None:
            # mask -> B x 35 (Max)
            loss_kl = loss_kl.masked_fill(~mask.bool(), 0.)
            l = mask.sum()

        if l is not None:
            return loss_kl.sum() / l

        return loss_kl.mean()

    
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
    
    def global_pairwise_contrastive_loss(self, a, b):
        labels = torch.eye(a.shape[0]).to(a.device)
        logits = a @ b.t() / self.global_temp
        print("-"*100)
        print("logits", logits.size())
        print(logits * self.global_temp)
        print("-"*100)
        print("prob", logits.size())
        print(logits.softmax(-1))
        print("-"*100)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = torch.sum(-log_prob*labels, dim=-1)
        return loss.mean()
    
    def cosine_loss(self, a, b):
        return (1-torch.sum(a*b, dim=-1)).mean()      
    
    @torch.no_grad()    
    def freeze(self):
        for model_pair in self.model_pairs:           
            for param in model_pair.parameters():
                param.requires_grad = False    
        


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