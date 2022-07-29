from glob import glob
from random import sample
import torch.nn.functional as F
from torch import nn
from transformers import  MaskFormerForInstanceSegmentation


class MaskformerPretrained(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self):
        super().__init__()
        self.model =  MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade") 
        self.activation = nn.Sigmoid()
        self.linear_output = nn.Linear(100, 1)

    def forward(self, x):
        out = self.model(x) 
        # Note taking last layer works better than adding a Linear Layer 
        out = out.masks_queries_logits

        # Linear Layer
        #out = out.permute(0,2,3,1)
        #out = self.linear_output(out)
        #out = out.permute(0,3, 1,2)

        # Take last hidden output
        out = out[:,-1,:,:].unsqueeze(1)
        out = F.interpolate(out, size = x.shape[2:], mode='bilinear', align_corners=True)
        out = self.activation(out)
        return out.squeeze()

