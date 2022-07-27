from PIL import Image
from transformers import SegformerFeatureExtractor
from matplotlib import pyplot as plt
import numpy as np
import torch

image = Image.open("data/training/images/satimage_117.png").convert("RGB")
segmentation = Image.open("data/training/groundtruth/satimage_117.png") #.convert("RGB")

image.show()
segmentation.show()

extractor = SegformerFeatureExtractor(reduce_labels=False, size=400, do_normalization=False)

encoded_inputs = extractor(image, segmentation, return_tensors="pt")

pixel_values = encoded_inputs["pixel_values"]
label_values = encoded_inputs["labels"]


print("pixel values ", pixel_values.shape , "label values ", label_values.shape)

pixel_values = pixel_values.squeeze_()
label_values = label_values.squeeze_()


print("pixel values ", pixel_values.shape , "label values ", label_values.shape)


pixel_values = torch.moveaxis(pixel_values, 0, -1)
#label_values = torch.moveaxis(label_values, 0, -1)

print("pixel values ", pixel_values.shape , "label values ", label_values.shape)

f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(pixel_values, interpolation='nearest')
f.add_subplot(1,2, 2)
plt.imshow(label_values, interpolation='nearest')
plt.show(block=True)



f.show()