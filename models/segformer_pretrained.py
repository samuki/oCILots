from glob import glob
from random import sample
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation
import json

""" 
This is an implementtion of the Segformer model from the following paper:

The implementation uses the HuggingFace transformer library providing a pretrained model.
We have tested different model architectures but ultimately we have found that nvidia/mit-b5 is the best model.

Please see the following link for more information:
https://huggingface.co/docs/transformers/model_doc/segformer 
"""


class SegFormerPretrained(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x).logits
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=True)
        out = self.activation(out)
        return out
