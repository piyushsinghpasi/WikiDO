CUDA_VISIBLE_DEVICES="2,4" WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=2  train_with_negatives_curriculum.py --method baseline_negatives --run_name temp_div_init0.5_mn.1_mx1_negatives_10_E3_saveall_correct_train --config ./configs/retrieval_wido_neg_cont.yaml &&


CUDA_VISIBLE_DEVICES="2,4" WANDB_MODE=online python -m torch.distributed.run --nproc_per_node=2  train_with_negatives_curriculum.py --method baseline_negatives --run_name temp_div_init0.5_mn.1_mx1_negatives_5_E3_saveall_correct_train --config ./configs/retrieval_wido_neg_cont_5.yaml
