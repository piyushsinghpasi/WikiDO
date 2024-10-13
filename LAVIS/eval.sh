#!/bin/bash

# BLIP2 VIT L WIKIDO WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_large_wikido.yaml

# BLIP2 VIT L WIKIDO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_large_coco_flickr.yaml

# BLIP2 VIT L COCO WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/coco_large_wikido.yaml

# BLIP2 VIT L FLICKR WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/flickr_large_wikido.yaml

# BLIP2 VIT L COCO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/coco_large_coco_flickr.yaml

# BLIP2 VIT L FLICKR COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/flickr_large_coco_flickr.yaml

#############################

# BLIP2 VIT G WIKIDO WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_giant_wikido.yaml

# BLIP2 VIT G WIKIDO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_giant_coco_flickr.yaml

# BLIP2 VIT G COCO WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/coco_giant_wikido.yaml

# BLIP2 VIT G FLICKR WIKIDO

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/flickr_giant_wikido.yaml

# BLIP2 VIT G COCO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/coco_giant_coco_flickr.yaml

# BLIP2 VIT G FLICKR COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/flickr_giant_coco_flickr.yaml


