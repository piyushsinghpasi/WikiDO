# original caption
CUDA_VISIBLE_DEVICES="1,2,4,5"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/clip/train_ft_b.yaml

# seed 2024 (feel free to change in config)
CUDA_VISIBLE_DEVICES="1,2,4,5"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/clip/train_ft_b_seed_2024.yaml

# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/retrieval_coco_ft.yaml


# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/clip/train_ft_b.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/retrieval_flickr_vitG_ft.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/retrieval_flickr_vitL_ft.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/retrieval_coco_ft.yaml
# CUDA_VISIBLE_DEVICES="0,1,2,3"  python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/vitL_coco_ft.yaml

