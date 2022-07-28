import os
import cv2
import numpy as np
import torch
import utils
import config
import glob
import albumentations as album
import torchvision

class FlexibleDataset(torch.utils.data.Dataset):
    # dataset to load images of varying sizes
    def __init__(
        self, path, device, use_patches=True, resize_to=(400, 400), augmentation=None
    ):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.augmentation = augmentation
        self.x = [
            os.path.join(self.path, "images", image_id)
            for image_id in sorted(os.listdir(os.path.join(self.path, "images")))
        ]
        self.y = [
            os.path.join(self.path, "groundtruth", image_id)
            for image_id in sorted(os.listdir(os.path.join(self.path, "groundtruth")))
        ]
        self.n_samples = len(self.x)

    def __getitem__(self, item):
        image = utils.load_from_path(self.x[item])[:, :, :3]
        mask = utils.load_from_path(self.y[item], grey=True)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        if self.use_patches:  # split each image into patches
            image, mask = utils.flexible_to_patches(image, mask)
        return utils.np_to_tensor(
            np.moveaxis(image, -1, 0).astype("float32"), self.device
        ), utils.np_to_tensor(mask.astype("float32"), self.device)

    def __len__(self):
        return self.n_samples


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = utils.load_all_from_path(os.path.join(self.path, "images"))[
            :, :, :, :3
        ]
        self.y = utils.load_all_from_path(os.path.join(self.path, "groundtruth"))
        if self.use_patches:  # split each image into patches
            self.x, self.y = utils.image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack(
                [cv2.resize(img, dsize=self.resize_to) for img in self.x], 0
            )
            self.y = np.stack(
                [cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0
            )
        self.x = np.moveaxis(
            self.x, -1, 1
        )  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def __getitem__(self, item):
        image = self.x[item]
        mask = self.y[item]
        return utils.np_to_tensor(image, self.device), utils.np_to_tensor(
            mask, self.device
        )

    def __len__(self):
        return self.n_samples


def load_test_data():
    test_path = config.TEST_PATH
    # predict on test set
    test_filenames = glob.glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    size = test_images.shape[1:3]
    test_images = np.stack(
        [cv2.resize(img, dsize=(config.HEIGHT, config.WIDTH)) for img in test_images], 0
    )
    test_images = test_images[:, :, :, :3]
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), config.DEVICE)
    transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    test_images = transform(test_images)
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    
    return test_images, size, test_filenames


def training_augmentation():
    train_transform = [
        album.RandomCrop(height=config.HEIGHT, width=config.WIDTH, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=config.p_augment),
        album.OneOf([
                album.MotionBlur(p=1),
                album.MedianBlur(blur_limit=3, p=1),
                album.Blur(blur_limit=3, p=1),
                
        ], p=config.p_augment),
        album.OneOf([
                album.OpticalDistortion(p=1),
                album.GridDistortion(p=1),
                album.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
        ], p=config.p_augment),
        
        album.OneOf([
            album.RandomContrast(limit=.6, p=1),
            album.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            album.RandomBrightness(limit=0.2, p=1)
        ], p=config.p_augment),
        album.RandomBrightnessContrast(p=config.p_augment),    
        album.RandomGamma(p=config.p_augment)
    ]
    return album.Compose(train_transform)


def validation_augmentation():
    test_transform = [
        album.CenterCrop(height=config.MINHEIGHT, width=config.MINWIDTH, always_apply=True),
    ]
    return album.Compose(test_transform)
