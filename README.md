# WIKIDO Cross-Modal Retrieval Challenge

This repository contains code and models for [WikiDO: A New Benchmark Evaluating Cross-Modal Retrieval for Vision-Language Models
](https://openreview.net/forum?id=EqaSEbU4LP#discussion). We also hosted a challenge inviting participants to evaluate and improve vision-language models (VLMs) on out-of-distribution (OOD) image-text retrieval tasks using the [WIKIDO Benchmark](https://huggingface.co/datasets/Pavankalyan/WikiDO).

## Challenge Overview

The WIKIDO Cross-Modal Retrieval Challenge aims to push the boundaries of vision-language models by testing their generalization capabilities beyond traditional datasets like MSCOCO and Flickr30K. The competition focuses on evaluating models on cross-modal retrieval tasks such as:

- **Image-to-Text Retrieval**
- **Text-to-Image Retrieval**

### Goal

The primary goal is to achieve high recall rates on OOD test sets that encompass domains and topics not seen during training. The competition seeks models that excel in zero-shot scenarios and significantly improve upon fine-tuning with the WIKIDO dataset.

## Dataset

The WIKIDO Benchmark consists of 384K diverse image-text pairs sourced from the Wikipedia Diversity Observatory. It provides a challenging environment to test the generalization capabilities of VLMs across unseen domains and topics. You can find the dataset on [Hugging Face](https://huggingface.co/datasets/Pavankalyan/WikiDO).

## Official Resources

- **Competition Homepage**: [WIKIDO Challenge](https://github.com/piyushsinghpasi/WikiDO)
- **Dataset**: [WIKIDO Benchmark on Hugging Face](https://huggingface.co/datasets/Pavankalyan/WikiDO)
- **Paper**: [NeurIPS24](https://github.com/piyushsinghpasi/WikiDO)
