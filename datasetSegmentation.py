import os
import cv2
import numpy as np
import torch
import utils
import config
import albumentations as album
from PIL import Image


class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
            self, feature_extractor, device, train=True):

        self.root_dir = "data"
        self.feature_extractor = feature_extractor
        self.train = train
        self.device = device

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, sub_path, "images" )
        self.ann_dir = os.path.join(self.root_dir, sub_path, "groundtruth")
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        #print("image file path ", os.path.join(self.img_dir, self.images[idx]))
        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx])) #.convert("RGBA")
      
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        #for k,v in encoded_inputs.items():
        #  encoded_inputs[k].squeeze_() # remove batch dimension
        image = encoded_inputs["pixel_values"].squeeze_()
        groundtruth = encoded_inputs["labels"].squeeze_()

        if len(groundtruth.shape) == 3:
            groundtruth = groundtruth[:, :, 0]

        groundtruth = torch.moveaxis(groundtruth, 0, -1)

        # print("image ", image.shape, " groundtruth ", groundtruth.shape)

        return image.type(torch.FloatTensor).to(self.device), groundtruth.type(torch.FloatTensor).to(self.device)
