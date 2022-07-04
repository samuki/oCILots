import os
import cv2
import numpy as np
import torch
import utils
import config
import albumentations as album


class FlexibleDataset(torch.utils.data.Dataset):
    # dataset to load images of varying sizes
    def __init__(self, path, device, use_patches=True, resize_to=(400, 400), augmentation=None, preprocessing=None, select_class_rgb_values=None):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_rgb_values=select_class_rgb_values        
        self.x = [os.path.join(self.path, 'images', image_id) for image_id in sorted(os.listdir(os.path.join(self.path, 'images')))]
        self.y = [os.path.join(self.path, 'groundtruth', image_id) for image_id in sorted(os.listdir(os.path.join(self.path, 'groundtruth')))]
        self.n_samples = len(self.x)
        
        
    def __getitem__(self, item):
        image = cv2.cvtColor(cv2.imread(self.x[item]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.y[item]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        if self.use_patches:  # split each image into patches
            image, mask = utils.image_to_patches(image, mask)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return utils.np_to_tensor(image, self.device), utils.np_to_tensor(mask, self.device)
    
    
    def __len__(self):
        return self.n_samples
        
class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400), augmentation=None, preprocessing=None, select_class_rgb_values=None):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x, self.y, self.n_samples = None, None, None
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_rgb_values=select_class_rgb_values
        self._load_data()
        
        
    def _load_data(self):  # not very scalable, but good enough for now
        self.x = utils.load_all_from_path(os.path.join(self.path, 'images'))[:,:,:,:3]
        self.y = utils.load_all_from_path(os.path.join(self.path, 'groundtruth'))
        if self.use_patches:  # split each image into patches
            self.x, self.y = utils.image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    
    def __getitem__(self, item):
        image = self.x[item]
        mask = self.y[item]
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return utils.np_to_tensor(image, self.device), utils.np_to_tensor(mask, self.device)
    
    def __len__(self):
        return self.n_samples


def _preprocess(preprocessing_fn=None):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        _transform = []
        if preprocessing_fn:
            _transform.append(album.Lambda(image=preprocessing_fn))
        _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))            
        return album.Compose(_transform)


def _flexible_preprocess(preprocessing_fn=None):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        _transform = []
        if preprocessing_fn:
            _transform.append(album.Lambda(image=preprocessing_fn))
        _transform.append(album.Lambda(image=flexible_to_tensor, mask=flexible_to_tensor))            
        return album.Compose(_transform)


def _training_augmentation():
    train_transform = [    
        album.RandomCrop(height=config.HEIGHT, width=config.WIDTH, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def _validation_augmentation():   
    test_transform = [
        album.PadIfNeeded(min_height=config.MINHEIGHT, min_width=config.MINWIDTH, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)
    
def to_tensor(x, **kwargs):
    return x

def flexible_to_tensor(x, **kwargs):
    return np.squeeze(x.astype('float32').transpose(2, 0, 1))


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
