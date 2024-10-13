#!/bin/bash

echo "Starting evaluations"


# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/b_100k_base_cf.yaml --output_dir NIPS/evals_final --run_name b_100k_base_cf --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/b_100k_base_wikido.yaml --output_dir NIPS/evals_final --run_name b_100k_base_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/base_cf.yaml --output_dir NIPS/evals_final --run_name b_100k_base_cf --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/base_wikido.yaml --output_dir NIPS/evals_final --run_name b_100k_base_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/b_100k_large_cf.yaml --output_dir NIPS/evals_final --run_name b_100k_large_cf --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/b_100k_large_wikido.yaml --output_dir NIPS/evals_final --run_name b_100k_large_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/large_cf.yaml --output_dir NIPS/evals_final --run_name b_100k_large_cf --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/large_wikido.yaml --output_dir NIPS/evals_final --run_name b_100k_large_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/b_200k_large_wikido.yaml --output_dir NIPS/evals_final --run_name b_200k_large_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original.py --config ./configs/Eval_configs/wikido_full_large_wikido.yaml --output_dir NIPS/evals_final --run_name wikido_full_large_wikido --evaluate

# CUDA_VISIBLE_DEVICES="4,6" python -m torch.distributed.run --nproc_per_node=2 train_retrieval_blip_original_itc_only.py --config ./configs/Eval_configs/b_100k_large_wikido_itc.yaml --output_dir NIPS/evals_final --run_name b_100k_large_wikido_itc --evaluate

echo "evaluations completed"

