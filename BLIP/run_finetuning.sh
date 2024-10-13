#!/bin/bash

echo "Started training on BLIP-VIT Large on 100k"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 train_retrieval_blip_original.py --config ./configs/NIPS/b_100k_large.yaml --output_dir NIPS/b_100k_large --run_name b_100k_large

echo "Started training on BLIP-VIT Large on 200k"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 train_retrieval_blip_original.py --config ./configs/NIPS/b_200k_large.yaml --output_dir NIPS/b_200k_large --run_name b_200k_large

echo "Started training on BLIP-VIT Large on Full"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 train_retrieval_blip_original.py --config ./configs/NIPS/wikido_full_large.yaml --output_dir NIPS/wikido_full_large --run_name wikido_full_large

echo "Started training on BLIP-VIT Base on 100k"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.run --nproc_per_node=4 train_retrieval_blip_original.py --config ./configs/NIPS/b_100k_base.yaml --output_dir NIPS/b_100k_base --run_name b_100k_base

echo "Successfully completed all finetuning processes."