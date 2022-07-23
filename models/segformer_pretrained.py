from glob import glob
from random import sample
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation
import json


class SegFormerPretrained(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self):
        super().__init__()
        # define model
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                                num_labels=1, 
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        #out = F.normalize(x)
        out = self.model(x).logits 
        out = F.interpolate(out, size = x.shape[-2:], mode='bilinear', align_corners=True)
        out = self.activation(out)
        return out
