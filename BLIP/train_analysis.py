'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

from prettytable import PrettyTable 

def train(model, data_loader, optimizer, epoch, device, config, args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,metadata in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.method in ['baseline','vq','vq_itm_mom']:
            (image, caption, idx) = metadata
            image = image.to(device,non_blocking=True) 
        elif args.method in ["domain_codebooks","separate_domain_codebooks","multimodal_codebooks", "simple_codebooks"]:
            (image, caption, idx, topic) = metadata
            image = image.to(device, non_blocking=True)
        else:
            (pos_image, pos_caption, idx, neg_images, neg_captions) = metadata
            pos_image = pos_image.to(device,non_blocking=True) 
          
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
        if args.method == 'baseline_negatives':
            loss_i2t, loss_t2i, loss_itm = model(pos_image, pos_caption, alpha=alpha, idx=idx, neg_images = neg_images, neg_captions = neg_captions)
        elif args.method == 'baseline':
            loss_i2t, loss_t2i, loss_itm = model(image, caption, alpha=alpha, idx=idx) 
        elif args.method in ['vq','vq_itm_mom']:
            loss_i2t, loss_t2i, loss_itm, loss_diversity = model(image, caption, alpha=alpha, idx=idx)
            diveristy_alpha = 0.2 #wav2vec2.0
        elif args.method in ["domain_codebooks","separate_domain_codebooks","multimodal_codebooks", "simple_codebooks"]:
            loss_i2t, loss_t2i, loss_itm = model(image, caption, alpha=alpha, idx=idx, topic=topic)
        
        loss_ita = (loss_i2t + loss_t2i)/2                 
        loss = loss_ita*0.2 + loss_itm
        
        if args.method in ['vq','vq_itm_mom']:
            loss = loss + (diveristy_alpha*loss_diversity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_i2t.item())
        metric_logger.update(loss_ita=loss_t2i.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.method in ['vq','vq_itm_mom']:
            wandb_stats = {
            "our_train_loss_itm": loss_itm.item(),
            "our_train_loss_ita": loss_ita.item(),
            "our_train_loss_i2t": loss_i2t.item(),
            "our_train_loss_t2i": loss_t2i.item(),
            "diversity_loss": loss_diversity.item()
            }
        else:
            wandb_stats = {
            "our_train_loss_itm": loss_itm.item(),
            "our_train_loss_ita": loss_ita.item(),
            "our_train_loss_i2t": loss_i2t.item(),
            "our_train_loss_t2i": loss_t2i.item(),
            }
        
        wandb.log(wandb_stats)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def min_max_mean(x, dim):
    """_summary_

    Args:
        x (_type_): _description_
        dim (_type_): deprecated
    """
    x = x.squeeze()
    print("std")
    print(x.std(dim=-1))
    print("Min")
    print(x.min(dim=-1)[0].view(-1))
    print(x.min(dim=-1)[1].view(-1))
    print("std")
    print(x.min(dim=-1)[0].view(-1).std(dim=-1))
    print("-"*100)
    print("Max")
    print(x.max(dim=-1)[0].view(-1))
    print(x.max(dim=-1)[1].view(-1))
    print("std")
    print(x.max(dim=-1)[0].view(-1).std(dim=-1))
    print("-"*100)
    print("Mean on a codeword across batch")
    print(x.mean(dim=-2))
    print("std")
    print(x.mean(dim=-2).std(dim=-1))
    

@torch.no_grad()
def evaluation(model, data_loader, device, config, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    
    image_feats = []
    image_embeds = []
    topics = []
    img_paths = []
    captions = []
    for idx, (image, img_id, img_path, caption, topic) in enumerate(data_loader): 
        text_input = model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = model.text_proj(text_output.last_hidden_state[:,0,:])
        
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])   
        img_paths.extend(img_path) 
        captions.extend(caption)   
        if args.method in ['vq','vq_itm_mom']:
            image_embed = model.image_quantizer(image_embed)["x"]
        elif args.method == "domain_codebooks":
            input_domain_tokens = torch.arange(0, 50, 1).to(image.device)
            output_domain_rep = []
            for top in topic:
                #top = 'monuments_and_buildings'
                output_domain_rep.append(model.codebooks[top](input_domain_tokens))
            domain_rep = torch.stack(output_domain_rep) 
            image_feat_domain,_ = model.image_domain_attention(image_embed.unsqueeze(1),domain_rep,domain_rep)
            image_embed = model.image_norm(image_feat_domain.squeeze(1) + image_embed)     
            text_feat_domain,_ = model.text_domain_attention(text_embed.unsqueeze(1),domain_rep,domain_rep)
            text_embed = model.image_norm(text_feat_domain.squeeze(1) + text_embed)
        elif args.method == "separate_domain_codebooks":
            input_domain_tokens = torch.arange(0, 50, 1).to(image.device)
            image_output_domain_rep = []
            text_output_domain_rep = []
            for top in topic:
                top = 'monuments_and_buildings'
                image_output_domain_rep.append(model.image_codebooks[top](input_domain_tokens))
                text_output_domain_rep.append(model.text_codebooks[top](input_domain_tokens))
            image_domain_rep = torch.stack(image_output_domain_rep)
            text_domain_rep = torch.stack(text_output_domain_rep) 
            image_feat_domain,_ = model.image_domain_attention(image_embed.unsqueeze(1),image_domain_rep,image_domain_rep)
            image_embed = model.image_norm(image_feat_domain.squeeze(1) + image_embed)     
            text_feat_domain,_ = model.text_domain_attention(text_embed.unsqueeze(1),text_domain_rep,text_domain_rep)
            text_embed = model.text_norm(text_feat_domain.squeeze(1) + text_embed)
        elif args.method == "multimodal_codebooks":
            input_domain_tokens = torch.arange(0, model.entries, 1).to(image.device)
            output_domain_rep = []
            for top in topic:
                output_domain_rep.append(model.codebooks[top](input_domain_tokens))
            
            domain_rep = torch.stack(output_domain_rep) 

            image_feat_domain, img_codeword_attn_wts = model.image_domain_attention(image_embed.unsqueeze(1),domain_rep,domain_rep)
            text_feat_domain, text_codeword_attn_wts = model.text_domain_attention(text_embed.unsqueeze(1),domain_rep,domain_rep)

        elif args.method == "simple_codebooks":
            input_domain_tokens = torch.arange(0, model.entries).unsqueeze(0).expand(len(topic), -1).to(image.device)
            domain_rep = model.codebooks(input_domain_tokens)

            image_feat_domain, img_codeword_attn_wts = model.image_domain_attention(image_embed.unsqueeze(1),domain_rep,domain_rep)
            text_feat_domain, text_codeword_attn_wts = model.text_domain_attention(text_embed.unsqueeze(1),domain_rep,domain_rep)
        
        if idx < 3: continue
        print("="*100)
        print("image_feat_domain")
        print(image_feat_domain)
        print(image_feat_domain.size())
        print("-"*100)
        min_max_mean(image_feat_domain, dim=-2)
        print("="*100)
        print("text_feat_domain")
        print(text_feat_domain)
        print(text_feat_domain.size())
        print("-"*100)
        min_max_mean(text_feat_domain, dim=-2)
        print("="*100)
        print("img_codeword_attn_wts")
        print(img_codeword_attn_wts)
        print(img_codeword_attn_wts.size())
        print("-"*100)
        min_max_mean(img_codeword_attn_wts, dim=-2)
        print("="*100)
        print("text_codeword_attn_wts")
        print(text_codeword_attn_wts)
        print(text_codeword_attn_wts.size())
        print("-"*100)
        min_max_mean(text_codeword_attn_wts, dim=-2)
        print("="*100)
        exit()
        # text_embed = model.text_norm(image_feat_domain.squeeze(1) + text_embed)
        # image_embed = model.image_norm(text_feat_domain.squeeze(1) + image_embed)

        text_embed = F.normalize(text_embed, dim=-1)
        image_embed = F.normalize(image_embed,dim=-1)      
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        topics.extend(topic)
        
    topic_map = {
        "monuments_and_buildings": 1, "industry": 2, "glam": 3,
    }
    topic_map_r = {
        v:k for k,v in topic_map.items()
    }

    topics = [topic_map.get(t, 0) for t in topics]
    topics = torch.Tensor(topics).long()

    text_embeds = torch.cat(text_embeds,dim=0)
    
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)

    print("Before")
    print(image_embeds.size())
    print(text_embeds.size())
    indices = torch.where((topics == 1) | (topics==2) | (topics==3))
    image_embeds = image_embeds[indices]
    text_embeds = text_embeds[indices]
    topics = topics[indices]

    img_paths = np.array(img_paths)
    img_paths = img_paths[indices]
    img_paths = np.char.replace(img_paths, "/workspace/", "./")
        
    captions = np.array(captions)
    captions = captions[indices]

    print("\nAfter")
    print(image_embeds.size())
    print(text_embeds.size())

    img_sims_matrix = image_embeds @ image_embeds.t()
    text_sims_matrix = text_embeds @ text_embeds.t()

    it_sims_matrix = image_embeds @ text_embeds.t()
    ti_sims_matrix = it_sims_matrix.t()


    top = 2
    num_samples = 1000
    idx = img_sims_matrix.argsort(dim=-1, descending=True)
    rows = torch.randint(0, len(img_sims_matrix), size=(num_samples,))
    rows = torch.arange(len(img_sims_matrix))
    idx = idx[rows, :top]
    idx = idx.cpu().long()

    text_sims = text_sims_matrix[rows.unsqueeze(1), idx]
    img_sims = img_sims_matrix[rows.unsqueeze(1), idx]

    it_sims_matrix = it_sims_matrix[rows.unsqueeze(1), idx]
    ti_sims_matrix = ti_sims_matrix[rows.unsqueeze(1), idx]

    # gap between top 2 imgs cant be more than this
    img_thres = 0.3 
    # gap between corresponding text must be greater than
    text_thres = 0.4

    matches = 0
    for idx_, rows_, img_path, caption, topic, text_sim, img_sim, img_text_sim, text_img_sim in \
        zip(idx, rows, img_paths[idx], captions[idx], topics[idx], text_sims, img_sims, it_sims_matrix, ti_sims_matrix):
        # print(abs(img_sim[0] - img_sim[1]))
        # print(abs(text_sim[0] - text_sim[1]))
        if topic[0] != topic[1] and \
            abs(img_sim[0] - img_sim[1]) < img_thres and \
            abs(text_sim[0] - text_sim[1]) > text_thres :
            matches += 1
            print("-"*100)
            print("idx")
            print(idx_)
            print("-"*100)
            print("rows")
            print(rows_)
            print("-"*100)
            print("img_paths")
            print(img_path)
            print("-"*100)
            print("captions")
            print(caption)
            print("-"*100)
            print("topics") 
            print(topic_map_r.get(topic[0].item()), topic_map_r.get(topic[1].item()))
            print("-"*100)

            sim_table = PrettyTable(['ref']+[f"img_{i}" for i in range(1, top+1)]+[f"text_{i}" for i in range(1, top+1)]) 

            top_half = torch.cat([img_sim, img_text_sim], dim=-1)
            bottom_half = torch.cat([text_img_sim, text_sim], dim=-1)
            
            sim_table.add_row(['img_1']+top_half.tolist())
            sim_table.add_row(['text_1']+bottom_half.tolist())


            print(sim_table)

            print()
            print("="*100)
            print()

    print(f"{matches}/{len(rows)} ({(matches*100/len(rows)):.2f}%) matches found ")
    exit()

    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in tqdm(enumerate(metric_logger.log_every(sims_matrix[start:end], 1000, header))):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        if args.method == 'vq_itm_mom':
            output = model.text_encoder_itm(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        else:
            output = model.text_encoder(text_ids[topk_idx], 
                                        attention_mask = text_atts[topk_idx],
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                    )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in tqdm(enumerate(metric_logger.log_every(sims_matrix[start:end], 1000, header))): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        if args.method == 'vq_itm_mom':
            output = model.text_encoder_itm(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        else:
            output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                        attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,                             
                                        return_dict = True,
                                    )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)        
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
    if args.method == 'baseline':
        from models.blip_retrieval import blip_retrieval
    elif args.method == 'baseline_negatives':
        from models.blip_retrieval_negatives import blip_retrieval
    elif args.method == 'vq':
        from models.blip_retrieval_vectorq import blip_retrieval
    elif args.method == "vq_itm_mom":
        from models.blip_retrieval_vq_itm_mom import blip_retrieval
    elif args.method == "domain_codebooks":
        from models.blip_retrieval_codebook import blip_retrieval
    elif args.method == "separate_domain_codebooks":
        from models.blip_retrieval_separate_codebook import blip_retrieval
    elif args.method == "multimodal_codebooks":
        from models.blip_retrieval_multimodal_codebook import blip_retrieval
    elif args.method == "simple_codebooks":
        from models.blip_retrieval_simple_codebook import blip_retrieval
    
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)   
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch, device, config, args)  
            
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config, args)
        #score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, args)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)
                                
            if val_result['r_mean']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = val_result['r_mean']        
                best_epoch = epoch  
                
                #test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                #print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             #**{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             #**{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   

            wandb.log(log_stats)
                    
        if args.evaluate: 
            break

        #dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--method', default='baseline')
    
    #Added arguments
    parser.add_argument('--run_name', help="wandb run name", type=str)
    
    args = parser.parse_args()
    
    wandb.init(project="smallest_train_curriculum", group=args.run_name)
    os.environ["WANDB_RUN_GROUP"] = f"{args.run_name}-" + wandb.util.generate_id()
    
    yaml = YAML()

    config = yaml.load(open(args.config, 'r'))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)