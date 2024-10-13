#!/bin/bash

# BLIP2 ITC+ITM VIT L WIKIDO WIKIDO

CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_large_wikido_itc_itm.yaml

# BLIP2 ITC VIT L WIKIDO COCO-FLICKR

CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/100_k_large_wikido_itc.yaml

