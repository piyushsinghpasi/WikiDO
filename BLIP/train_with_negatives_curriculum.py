'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

# CUDA_VISIBLE_DEVICES=1,2,4,5 python -m torch.distributed.run --nproc_per_node=4  train_with_negatives_curriculum.py --output_dir output_curriculum/epoch3 --method baseline_negatives --config ./configs/retrieval_wido_neg_cont.yaml --run_name epochthres3 
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


def train(model, data_loader, optimizer, epoch, device, config, args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_i2t', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_t2i', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,metadata in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.method in ['baseline','vq','vq_itm_mom']:
            (image, caption, idx) = metadata
            image = image.to(device,non_blocking=True) 
        else:
            (pos_image, pos_caption, idx, neg_images, neg_captions) = metadata
            pos_image = pos_image.to(device,non_blocking=True) 
            neg_images = neg_images.to(device,non_blocking=True) 
          
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
        if args.method == 'baseline_negatives':
            loss_i2t, loss_t2i, loss_itm = model(
                                            pos_image, 
                                            pos_caption, 
                                            alpha=alpha, 
                                            idx=idx, 
                                            neg_images = neg_images, 
                                            neg_captions = neg_captions, 
                                            epoch=epoch,
                                            epoch_threshold = config.get('epoch_threshold', -100),
                                            step = (epoch*len(data_loader) + i + 1),
                                            )
        elif args.method == 'baseline':
            loss_i2t, loss_t2i, loss_itm = model(image, caption, alpha=alpha, idx=idx) 
        elif args.method in ['vq','vq_itm_mom']:
            loss_i2t, loss_t2i, loss_itm, loss_diversity = model(image, caption, alpha=alpha, idx=idx)
            diveristy_alpha = 0.1 #wav2vec2.0
        
        loss_ita = (loss_i2t + loss_t2i)/2                 
        loss = loss_ita + loss_itm
        
        if args.method in ['vq','vq_itm_mom']:
            loss = loss + (diveristy_alpha*loss_diversity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_i2t=loss_i2t.item())
        metric_logger.update(loss_t2i=loss_t2i.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.method in ['vq','vq_itm_mom']:
            wandb_stats = {
            "our_train_loss_itm": loss_itm.item(),
            "our_train_loss_ita": loss_ita.item(),
            "our_train_loss_i2t": loss_i2t.item(),
            "our_train_loss_t2i": loss_t2i.item(),
            "diversity_loss": loss_diversity.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "temperature": model.module.temp.item(),
            }
        else:
            wandb_stats = {
            "our_train_loss_itm": loss_itm.item(),
            "our_train_loss_ita": loss_ita.item(),
            "our_train_loss_i2t": loss_i2t.item(),
            "our_train_loss_t2i": loss_t2i.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "temperature": model.module.temp.item(),
            }
        
        wandb.log(wandb_stats)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}  


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
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    # print('text_embeds',{text_embeds.shape})

    
    image_feats = []
    image_embeds = []
    for image, img_id, img_path in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])       
        if args.method in ['vq','vq_itm_mom']:
            image_embed = model.image_quantizer(image_embed)["x"]     
        image_embed = F.normalize(image_embed,dim=-1)      
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)

    # print('image_embeds',{image_embeds.shape})
    # print('image_feats',{image_feats.shape})
    
    sims_matrix = image_embeds @ text_embeds.t()

    return sims_matrix

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
def itc_eval(sims_matrix, txt2img, img2txt, device):
    # print(img2txt)
    # print(sims_matrix)
    # print(sims_matrix.shape)
    # print(f'Max',torch.max(sims_matrix, dim = -1))
    # print(f'Min',torch.min(sims_matrix, dim = -1))
    # print(f'Mean',torch.mean(sims_matrix, dim = -1))
    
    img_2_txt = [img2txt[i] for i in range(len(sims_matrix))]
    img_2_txt = torch.Tensor(img_2_txt).long().to(device)
    img2text_pred_rank = sims_matrix.argsort(dim=-1, descending=True) # max `first`

    batch_size = 256

    img2text_ranks = []
    for i in range(0, len(img2text_pred_rank), batch_size): 
        img2text_ranks.append(torch.where(img2text_pred_rank[i:i+batch_size,:] == img_2_txt[i:i+batch_size,:])[1].unsqueeze(-1))
        # print(f'img2text_pred_rank',img2text_pred_rank[i:i+batch_size,:].shape)
        # print(f'img_2_txt',img_2_txt[i:i+batch_size,:].shape)
        # print(torch.where(img2text_pred_rank[i:i+batch_size,:] == img_2_txt[i:i+batch_size,:])[1].unsqueeze(-1).shape)
    # img2text_ranks.append(torch.nonzero(img2text_pred_rank[i:i+batch_size,:] == img_2_txt[i:i+batch_size,:], as_tuple=True)[1])

    img2text_ranks = torch.cat(img2text_ranks, dim = 0)
    # print(f'img2text_ranks',img2text_ranks.shape)

    txt_2_img = [txt2img[i] for i in range(len(sims_matrix))]
    txt_2_img = torch.Tensor(txt_2_img).long().unsqueeze(-1).to(device)
    txt2img_pred_rank = sims_matrix.t().argsort(dim = -1, descending = True)

    txt2img_ranks = []
    for i in range(0, len(txt2img_pred_rank), batch_size):
        txt2img_ranks.append(torch.where(txt2img_pred_rank[i:i+batch_size,:] == txt_2_img[i:i+batch_size,:])[1].unsqueeze(-1))

    txt2img_ranks = torch.cat(txt2img_ranks, dim = 0)

    img2text_r1 = (img2text_ranks < 1).sum().item()
    img2text_r5 = (img2text_ranks < 5).sum().item()
    img2text_r10 = (img2text_ranks < 10).sum().item()

    txt2img_r1 = (txt2img_ranks < 1).sum().item()
    txt2img_r5 = (txt2img_ranks < 5).sum().item()
    txt2img_r10 = (txt2img_ranks < 10).sum().item()

    tr1 = 100.0 * img2text_r1 / len(img2text_ranks)
    tr5 = 100.0 * img2text_r5 / len(img2text_ranks)
    tr10 = 100.0 * img2text_r10 / len(img2text_ranks)

    ir1 = 100 * txt2img_r1 / len(txt2img_ranks)
    ir5 = 100 * txt2img_r5 / len(txt2img_ranks)
    ir10 = 100 * txt2img_r10 / len(txt2img_ranks)

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


    # print(f'img2text_r1',img2text_r1)
    # print(f'img2text_r5',img2text_r5)
    # print(f'img2text_r10',img2text_r10)


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
    
    model = blip_retrieval(
                pretrained=config['pretrained'], 
                image_size=config['image_size'], 
                vit=config['vit'], 
                vit_grad_ckpt=config['vit_grad_ckpt'], 
                vit_ckpt_layer=config['vit_ckpt_layer'], 
                queue_size=config['queue_size'], 
                negative_all_rank=config['negative_all_rank'],
                label_smoothing=config.get('label_smoothing', 0.),
                temperature_damp = config.get('temperature_damp', False),
                damp_factor = config.get('damp_factor', 0.98),
                warmup_step = config.get('warmup_step', 100),
    )

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
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
            
            ## Checking eval function
            train_stats = train(model, train_loader, optimizer, epoch, device, config, args)  
            
        sims_matrix = evaluation(model_without_ddp, val_loader, device, config, args)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, args)


        if utils.is_main_process():  
            # Replaced itm_eval with scores as input with itc_eval
            val_result = itc_eval(sims_matrix, val_loader.dataset.txt2img, val_loader.dataset.img2txt, device)  
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
                # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                # print(test_result)
                
            save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))  

            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                            #  **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                ## Removed train stats
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                            #  **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
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
    args.output_dir = os.path.join(config.get("output_dir", "./output"), args.run_name)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)