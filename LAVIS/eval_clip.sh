#!/bin/bash


# # CLIP VIT L Zshot WIKIDO

# CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/clip/eval_zshot_test_wikido_id_ood.yaml

# # CLIP VIT L Zshot COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/clip/eval_zshot_test_wikido_coco_flickr.yaml

# # CLIP VIT L WIKIDO WIKIDO

# CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/clip/eval_wikido_trained_test_wikido_id_ood.yaml

# # CLIP VIT L WIKIDO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="3,2,1,0" python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/clip/eval_wikido_trained_test_coco_flickr.yaml


# # ----------------------------------------------------------------------------------------------------

# # CLIP VIT L Flickr WIKIDO

# CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/clip/eval_flickr_trained_test_wikido.yaml

# # CLIP VIT L Flickr COCO-FLICKR

# CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/clip/eval_flickr_trained_test_cf.yaml

# # CLIP VIT L COCO WIKIDO

# CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/clip/eval_coco_trained_test_wikido.yaml

# # CLIP VIT L COCO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/clip/eval_coco_trained_test_cf.yaml

# ----------------------------------------------------------------------------------------------------

# # CLIP VIT L Flickr WIKIDO

# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/clip/eval_orig_cap_wikido_trained_test_wikido_id_ood.yaml

# # CLIP VIT L Flickr COCO-FLICKR

# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/clip/eval_orig_cap_wikido_trained_test_coco_flickr.yaml

# # CLIP VIT L COCO WIKIDO

# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/clip/eval_wikido_trained_test_wikido_id_ood_seed.yaml

# # CLIP VIT L COCO COCO-FLICKR

# CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/clip/eval_wikido_trained_test_coco_flickr_seed.yaml


# CLIP VIT L Flickr WIKIDO

CUDA_VISIBLE_DEVICES="6,7" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/clip/eval_orig_cap_zshot_test_wikido_id_ood.yaml
