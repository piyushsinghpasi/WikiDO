import json
from PIL import Image

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import torch.nn.functional as F 
import torch.optim.lr_scheduler as lr_scheduler
from utils import cosine_lr_schedule
import wandb
from alive_progress import alive_bar

from clip_eval import get_loader, evaluation, itm_eval

from clip_models import custom_clip

lr = 1e-6
min_lr = 1e-7

# run_name = f"clip_finetuning_baseline_b128_gradclip_iid_temp_0.5_lr_{lr}_weightDecay_0.001_logits_train_val_diff"
wandb_runname = "clip_100K"
output_dir = f"./clip-checkpoints/final/{wandb_runname}"
os.makedirs(output_dir, exist_ok=True)

wandb.init(project="NIPS_WIKIDO", name=wandb_runname)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_path)

data_path = "/workspace/pavan/multimodal/data/multimodal/final_splits/balanced_100k_train_42.json"
data_path_val = "/workspace/pavan/multimodal/data/multimodal/final_splits/val_3k_42.json"
img_base = "/workspace/pavan/multimodal/data/multimodal"

blip_config = "/workspace/pavan/multimodal/BLIP/configs/retrieval_coco_clip.yaml"

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    print(model)
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if isinstance(attr, nn.Linear):
                    attr.weight.data = attr.weight.data.half()
                    if attr.bias is not None:
                        attr.bias.data = attr.bias.data.half()
                elif attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
    print(model)


model = custom_clip(0.1, model_path)
# model = torch.nn.DataParallel(model)
model = model.to(device)
# convert_weights(model.clip_model)

with open(data_path, 'r') as f:
    all_data = json.load(f)

text_data = [data_point["caption"] for data_point in all_data]
img_pth = [os.path.join(img_base,data_point["image"][1:]) for data_point in all_data]
        
class custom_dataset():

    def __init__(self, img_path , txt_data):
        self.image_path = img_path
        self.text_data  = processor(text=txt_data, return_tensors="pt", padding=True, truncation=True)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = processor(images=Image.open(self.image_path[idx]).convert("RGB"), return_tensors="pt", padding=True)
        return image['pixel_values'], self.text_data['input_ids'][idx], self.text_data['attention_mask'][idx]


# def evaluate(data_path, img_base, img_batch_size, text_batch_size, device, model, processor):

#     with open(data_path, 'r') as f:
#         all_data = json.load(f)
        
#     image_paths = [os.path.join(img_base, data_point["image"][1:] if data_point["image"].startswith("/") else data_point["image"]) for data_point in all_data]
#     text_data = [data_point["caption"] for data_point in all_data]

#     img_batches = [image_paths[i: i+img_batch_size] for i in range(0, len(image_paths), img_batch_size)]

#     # img batch gold ranks
#     img_gold_rank = list(range(len(image_paths)))
#     img_gold_rank = [img_gold_rank[i: i+img_batch_size] for i in range(0, len(img_gold_rank), img_batch_size)]

#     text_batches = [text_data[i: i+text_batch_size] for i in range(0, len(text_data), text_batch_size)]

#     print(f'Total image batches: {len(img_batches)}')
#     print(f'Total text batches: {len(text_batches)}')

#     img_ranks = []
#     text_ranks = []

#     text2img_all_scores = None

#     img2text_r1 = 0.
#     img2text_r5 = 0.
#     img2text_r10 = 0.

#     text2img_r1 = 0.
#     text2img_r5 = 0.
#     text2img_r10 = 0.

#     model.eval()
#     with torch.no_grad():

#         with alive_bar(len(img_gold_rank), title="Eval ", length=80,
#                        spinner='waves', bar='halloween') as bar:
#             # iterate over all image batches
#             for idx, (img_batch_gold_rank, img_batch) in enumerate(zip(img_gold_rank, img_batches)):

#                 loaded_img_batch = [Image.open(x) for x in img_batch]
#                 processed_img_batch = processor(images=loaded_img_batch, return_tensors="pt", padding=True)
#                 processed_img_batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in processed_img_batch.items()}
                
#                 img2text_batch_scores = []
#                 text2img_batch_scores = []

#                 # iterate over all text batches
#                 for text_batch_idx, text_batch in enumerate(text_batches):
#                     processed_text_batch = processor(text=text_batch, return_tensors="pt", padding=True, truncation=True)
#                     processed_text_batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in processed_text_batch.items()}

#                     outputs = model(**processed_img_batch,      **processed_text_batch)
                    
#                     processed_text_batch = {k:(v.detach().cpu() if isinstance(v, torch.Tensor) else "") for k,v in processed_text_batch.items()}
#                     del processed_text_batch

#                     img2text_scores = outputs[0].detach().cpu() # text score for each image
#                     text2img_scores = outputs[1].detach().cpu() # image score for each text
                    
#                     img2text_batch_scores.append(img2text_scores)
#                     text2img_batch_scores.append(text2img_scores)

                            
#                 img2text_batch_scores = torch.hstack(img2text_batch_scores) # text scores for each image in batch

#                 text2img_batch_scores = torch.vstack(text2img_batch_scores) # curr image batch scores all text in whole data
                
#                 if text2img_all_scores is not None:
#                     text2img_all_scores = torch.hstack([text2img_all_scores, text2img_batch_scores])
#                 else:
#                     text2img_all_scores = text2img_batch_scores
                    
#                 # img rank calculation
#                 img2text_pred_rank = img2text_batch_scores.argsort(dim=-1, descending=True) # max `first`
#                 img2text_gold_rank = torch.Tensor(img_batch_gold_rank).long().unsqueeze(-1)
#                 img2text_ranks = torch.where(img2text_pred_rank == img2text_gold_rank)[1]
#                 img2text_r1 += (img2text_ranks < 1).sum().item()
#                 img2text_r5 += (img2text_ranks < 5).sum().item()
#                 img2text_r10 += (img2text_ranks < 10).sum().item()


#                 # progress_bar.update(1)
#                 bar()
                    
#     # image to text
                
#     img2text_r1 = img2text_r1 / len(image_paths)
#     img2text_r5 = img2text_r5 / len(image_paths)
#     img2text_r10 = img2text_r10 / len(image_paths)
#     img2text_mean = (img2text_r1 + img2text_r5 + img2text_r10) / 3

#     Results = {}

#     # Text to image
#     text_gold_ranks = torch.arange(len(text_data)).long().unsqueeze(-1)
#     text2img_pred_ranks = text2img_all_scores.argsort(dim=-1, descending=True)
#     text2img_ranks = torch.where(text2img_pred_ranks == text_gold_ranks)[1]

#     text2img_r1 = (text2img_ranks < 1).sum().item()
#     text2img_r5 = (text2img_ranks < 5).sum().item()
#     text2img_r10 = (text2img_ranks < 10).sum().item()


#     text2img_r1 = text2img_r1 / len(text_data)
#     text2img_r5 = text2img_r5 / len(text_data)
#     text2img_r10 = text2img_r10 / len(text_data)
#     text2img_mean = (text2img_r1 + text2img_r5 + text2img_r10)/3


#     r_mean = (text2img_mean+img2text_mean)/2
#     eval_result = {
#         "txt_r1": img2text_r1,
#         "txt_r5": img2text_r5,
#         "txt_r10": img2text_r10,
#         "txt_r_mean": img2text_mean,
#         "img_r1": text2img_r1,
#         "img_r5": text2img_r5,
#         "img_r10": text2img_r10,
#         "img_r_mean": text2img_mean,
#         "r_mean": r_mean,
#     }
#     wandb_stats = {
#         f"eval/{k}":v for k, v in eval_result.items()
#     }
#     wandb.log(wandb_stats)

#     print("\n".join([f"{k}: {(v*100):.2f}" for k,v in eval_result.items()]))

#     with open(os.path.join(output_dir,"evaluate.txt"), "a") as f:
#         f.write(json.dumps(eval_result) + "\n")

#     return eval_result

# config
batch_size = 64
num_epochs = 6

dataset = custom_dataset(img_pth,text_data)
train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

val_dataloader = get_loader(blip_config, split_type='val')

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
# scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.4, total_iters=6)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
best = 0
for epoch in range(num_epochs):

    cosine_lr_schedule(optimizer, epoch, num_epochs, lr, 0)

    print("Training..")
    total_loss = 0
    batch  = 0
    model.train()
    num_steps = len(train_dataloader.dataset) // batch_size
    with alive_bar(num_steps, title=f"Training Epoch: {epoch+1} ", length=80,
                   spinner='waves', bar='halloween') as bar:
        for idx, (images,texts,att_mask) in enumerate(train_dataloader):

            batch = batch+1
            optimizer.zero_grad()        
            images= images.to(device).squeeze(dim=1)
            texts = texts.to(device)
            att_mask = att_mask.to(device)

            if (idx == num_steps):
                break

            # Forward pass
            logits = model(input_ids = texts, pixel_values = images, attention_mask = att_mask)
            logits_per_image = logits[0]
            logits_per_text = logits[1]
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            
            # Compute loss
            train_loss_i2t = loss_img(logits_per_image,ground_truth)
            train_loss_t2i = loss_txt(logits_per_text,ground_truth)
            loss = (train_loss_i2t + train_loss_t2i)/2

            wandb_stats = {
                "train/train_loss_i2t": train_loss_t2i.item(),
                "train/train_loss_t2i": train_loss_t2i.item(),
                "train/train_loss_mean": loss.item(),
                "train/temperature" : model.temp.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            wandb.log(wandb_stats)

            total_loss = total_loss + loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # convert_models_to_fp32(model)

            optimizer.step()
            # convert_weights(model.clip_model)

            bar()

        print(f"Epoch: {epoch+1}/{num_epochs} Batch: {batch}/{len(train_dataloader)} Loss: {loss:.4f}")


    # Validation   
    print("Validation...")
    model.eval()
    # val_result = evaluate(data_path_val,img_base,batch_size,batch_size,device,model,processor)
    out = evaluation(model.clip_model, val_dataloader, device, load_clip_model=False)
    val_result = itm_eval(*out, val_dataloader.dataset.txt2img, val_dataloader.dataset.img2txt)

    wandb_stats = {
        f"eval/{k}":v for k, v in val_result.items()
    }
    wandb.log(wandb_stats)

    print("\n".join([f"{k}: {(v):.2f}" for k,v in val_result.items()]))

    if val_result['r_mean'] > best:

        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))  
        best = val_result['r_mean']        
        best_epoch = epoch  
        
    save_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_obj, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))  
  


