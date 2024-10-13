from ruamel.yaml import YAML
from data import create_dataset, create_sampler, create_loader
import torch
import torch.nn.functional as F
from tqdm import tqdm

config_path = "./configs/retrieval_wido_negative.yaml"
yaml = YAML()
config = yaml.load(open(config_path, 'r'))
train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)
samplers = [None, None, None]
train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   

method = "domain_codebooks"
print("Creating model")
if method == 'baseline':
    from models.blip_retrieval import blip_retrieval
elif method == 'baseline_negatives':
    from models.blip_retrieval_negatives import blip_retrieval
elif method == 'vq':
    from models.blip_retrieval_vectorq import blip_retrieval
elif method == "vq_itm_mom":
    from models.blip_retrieval_vq_itm_mom import blip_retrieval
elif method == "domain_codebooks":
    from models.blip_retrieval_codebook import blip_retrieval

model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
device = torch.device('cuda')
model = model.to(device)   



model.eval()
text_ids = []
text_embeds = []  
text_atts = []
image_feats = []
image_embeds = []
topics = []

data_loader = test_loader

# for image, img_id, img_path, caption, topic in tqdm(data_loader): 
#     text_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
#     text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
#     text_embed = model.text_proj(text_output.last_hidden_state[:,0,:])
    
#     image = image.to(device) 
#     image_feat = model.visual_encoder(image)   
#     image_embed = model.vision_proj(image_feat[:,0,:])       
#     if method in ['vq','vq_itm_mom']:
#         image_embed = model.image_quantizer(image_embed)["x"]
#     elif method == "domain_codebooks":
#         input_domain_tokens = torch.arange(0, 50, 1).to(image.device)
#         output_domain_rep = []
#         for top in topic:
#             #top = 'monuments_and_buildings'
#             output_domain_rep.append(model.codebooks[top](input_domain_tokens))
#         domain_rep = torch.stack(output_domain_rep) 
#         image_feat_domain,_ = model.image_domain_attention(image_embed.unsqueeze(1),domain_rep,domain_rep)
#         image_embed = model.image_norm(image_feat_domain.squeeze(1) + image_embed)     
#         text_feat_domain,_ = model.text_domain_attention(text_embed.unsqueeze(1),domain_rep,domain_rep)
#         text_embed = model.text_norm(text_feat_domain.squeeze(1) + text_embed)
#         input_domain_tokens = input_domain_tokens.to('cpu')
#         domain_rep = domain_rep.to('cpu')
#         image_feat_domain = image_feat_domain.to('cpu')
#         text_feat_domain = text_feat_domain.to('cpu')
    
#     text_embed = F.normalize(text_embed.squeeze(1), dim=-1)
#     image_embed = F.normalize(image_embed,dim=-1)      
# #     image_feats.append(image_feat.cpu())
#     image_embeds.append(image_embed.to('cpu'))
#     text_embeds.append(text_embed.to('cpu'))   
# #     text_ids.append(text_input.input_ids)
# #     text_atts.append(text_input.attention_mask)
#     topics.extend(topic)
#     torch.cuda.empty_cache()
    
# text_embeds = torch.cat(text_embeds,dim=0)
# #text_ids = torch.cat(text_ids,dim=0)
# #text_atts = torch.cat(text_atts,dim=0)
# #text_ids[:,0] = model.tokenizer.enc_token_id
    
# #image_feats = torch.cat(image_feats,dim=0)
# image_embeds = torch.cat(image_embeds,dim=0)

# # save the image_embeds, text_embeds, the corresponding topics in csv
# torch.save(text_embeds, "./text_embeds"+method+".pt")
# torch.save(image_embeds, "./image_embeds"+method+".pt")

topics = ["I", "Donno", "KNOW", "How"]

import numpy as np

text_embeds = torch.rand(4)
image_embeds = torch.rand(4)

torch.save(text_embeds, "./text_embeds_domain_codebook.pt")
torch.save(image_embeds, "./image_embeds_domain_codebook.pt")
np.savetxt("domain_codebook_topics.csv", topics, delimiter=", ", fmt='% s')