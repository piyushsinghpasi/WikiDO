import json
from PIL import Image

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import torch.nn.functional as F 
import torch.optim.lr_scheduler as lr_scheduler
from utils import cosine_lr_schedule


class custom_clip(nn.Module):

    def __init__(self, temp, model_path):
        super().__init__()    
        self.clip_model = CLIPModel.from_pretrained(model_path)
        self.temp = nn.Parameter(torch.Tensor([temp]))

    def forward(self, input_ids, pixel_values, attention_mask):
        output = self.clip_model(input_ids = input_ids, pixel_values = pixel_values, attention_mask = attention_mask)

        text_embeds = F.normalize(output.text_embeds, p=2, dim=-1)
        image_embeds = F.normalize(output.image_embeds, p=2, dim=-1)
        text2image_scores = text_embeds @ image_embeds.t()
        image2text_scores = text2image_scores.t()

        text2image_scores = text2image_scores / self.temp
        image2text_scores = image2text_scores / self.temp

        # return output.logits_per_image, output.logits_per_text
        return image2text_scores, text2image_scores