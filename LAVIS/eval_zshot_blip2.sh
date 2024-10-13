#!/bin/bash


CUDA_VISIBLE_DEVICES="6,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/zshot_giant_wikido.yaml

CUDA_VISIBLE_DEVICES="6,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/zshot_large_wikido.yaml

CUDA_VISIBLE_DEVICES="6,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/zshot_giant_coco_flickr.yaml

CUDA_VISIBLE_DEVICES="6,0" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip2/eval/zshot_large_coco_flickr.yaml


