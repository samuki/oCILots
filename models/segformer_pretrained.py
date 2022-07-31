from glob import glob
from random import sample
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import json
import torch
import config


def resize(input,size=None,scale_factor=None,mode='nearest'):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode)


class SegFormerPretrained(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self):
        super().__init__()
        # define model
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",num_labels=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x).logits 
        #out = resize(
        #    input=out,
        #    size=x.shape[2:],
        #    mode='bilinear')
        #out = self.activation(out)
        out = F.interpolate(out, size = x.shape[2:], mode='bilinear', align_corners=True)
        out = self.activation(out)
        return out