'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
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

from models.blip_ret_chan_bert import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_i2t, loss_t2i, loss_itm = model(image, caption, alpha=alpha, idx=idx) 
        loss_ita = (loss_i2t + loss_t2i)/2                 
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_i2t.item())
        metric_logger.update(loss_ita=loss_t2i.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
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


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    images = data_loader.dataset.image
    num_text = len(texts)
    text_bs = 64
    text_ids = []
    text_embeds = []
    text_atts = []
    cap_l = []
    
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        cap_lens = text_input.attention_mask.sum(-1).unsqueeze(0).long()[0]
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_token_output = model.token_encoder(text_input.input_ids, attention_mask = text_input.attention_mask)
        text_embed = model.text_proj(text_output.last_hidden_state[:,0,:])
        token_embed = model.text_proj(text_token_output.last_hidden_state[:,1:,:])
        text_embed = text_embed.unsqueeze(1).repeat(1,34,1)
        token_embed += text_embed
        token_embed = F.normalize(token_embed, dim=-1)
        text_embeds.append(token_embed)   
        #text_embeds.append(text_embed)
        a = text_input.input_ids
        a[:,0] = model.tokenizer.enc_token_id
        text_ids.append(a)
        text_atts.append(text_input.attention_mask)
        cap_l.append(cap_lens)
    
    image_feats = []
    image_embeds = []
    
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat)            
        image_embed = image_embed[:,1:,:] 
        cls_embed = image_embed[:,0,:]
        cls_embed = cls_embed.unsqueeze(1).repeat(1,256,1)
        image_embed += cls_embed
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
    
    sims_matrix = []

    for i in tqdm(range(len(image_embeds))):
        sims_row = []
        for j in range(len(text_embeds)):
            sims_col = model.get_score(image_embeds[i], text_embeds[j], cap_l[j], is_image_first=True)
            sims_row.append(sims_col)
        sims_matrix.append(torch.cat(sims_row,dim=1))
    sims_matrix = torch.cat(sims_matrix, dim=0)
    # sims_matrix = []
    
    # for i in tqdm(range(len(text_embeds))):
    #     sims_row = []
    #     for j in range(len(image_embeds)):
    #         sims_col = model.get_score(text_embeds[i], image_embeds[j], cap_l[i], is_image_first=False)
    #         sims_row.append(sims_col)
    #     sims_matrix.append(torch.cat(sims_row,dim=1))
    # sims_matrix = torch.cat(sims_matrix, dim=0)
    
    # sims_matrix = sims_matrix.t()

    
        
    
    
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    
    text_ids1 = torch.cat(text_ids,dim=0)
    text_atts1 = torch.cat(text_atts,dim=0)
    image_feats1 = torch.cat(image_feats,dim=0)
    
    print("start-1")
    print(sims_matrix.shape)

    for i,sims in tqdm(enumerate(metric_logger.log_every(sims_matrix[start:end], 1000, header))):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats1[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids1[topk_idx], 
                                    attention_mask = text_atts1[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    print("end-1")
        
    sims_matrix = []
    
    for i in tqdm(range(len(text_embeds))):
        sims_row = []
        for j in range(len(image_embeds)):
            sims_col = model.get_score(text_embeds[i], image_embeds[j], cap_l[i], is_image_first=False)
            sims_row.append(sims_col)
        sims_matrix.append(torch.cat(sims_row,dim=1))
    sims_matrix = torch.cat(sims_matrix, dim=0)
    #sims_matrix = sims_matrix.t()
    
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    

    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)   
    
    print("start-2") 
    
    for i,sims in tqdm(enumerate(metric_logger.log_every(sims_matrix[start:end], 1000, header))): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats1[topk_idx.cpu()].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids1[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts1[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim
        
    print("end-2")

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
            
            train_stats = train(model, train_loader, optimizer, epoch, device, config)  
        
        if epoch==5:    
            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
            
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
                    
                    test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                    print(test_result)
                
                if args.evaluate:                
                    log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},                  
                                }
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")     
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},  
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")   
                        
                wandb.log(log_stats)
        elif args.evaluate:
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)                
                log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")    
                wandb.log(log_stats)
                    
        if args.evaluate: 
            break

        dist.barrier()     
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
    
    #Added arguments
    parser.add_argument('--run_name', help="wandb run name", type=str)
    
    args = parser.parse_args()
    
    wandb.init(project="BLIP_New", name=args.run_name)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)