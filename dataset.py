import os
import cv2
import numpy as np
import torch
import utils
import glob
import albumentations as album
import torchvision
from PIL import Image
from config import config


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
        mask = utils.load_from_path(self.y[item])
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if self.use_patches:  # split each image into patches
            image, mask = utils.image_to_patches(image, mask)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image = utils.np_to_tensor(
            np.moveaxis(image, -1, 0).astype("float32"), self.device
        )

        if config.USE_NORMALIZATION:

            transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            image = transform(image)

        return image, utils.np_to_tensor(mask.astype("float32"), self.device)

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
        tensor = utils.np_to_tensor(image, self.device), utils.np_to_tensor(
            mask, self.device
        )

        if config.USE_NORMALIZATION:

            transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            tensor = transform(tensor)

        return tensor

    def __len__(self):
        return self.n_samples


class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """Image (semantic) segmentation dataset.
    This has been tested using the SegFormer feature extractor from the HuggingFace library.
    """

    def __init__(self, feature_extractor, device, train=True):

        self.root_dir = "data"
        self.feature_extractor = feature_extractor
        self.train = train
        self.device = device

        sub_path = "trainingEth" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
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

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        segmentation_map = Image.open(
            os.path.join(self.ann_dir, self.annotations[idx])
        )  # .convert("RGBA")

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt"
        )

        # for k,v in encoded_inputs.items():
        #  encoded_inputs[k].squeeze_() # remove batch dimension
        image = encoded_inputs["pixel_values"].squeeze_()
        groundtruth = encoded_inputs["labels"].squeeze_()

        if len(groundtruth.shape) == 3:
            groundtruth = groundtruth[:, :, 0]

        return image.type(torch.FloatTensor).to(self.device), groundtruth.type(
            torch.FloatTensor
        ).to(self.device)


def load_test_data():
    test_path = config.TEST_PATH
    # predict on test set
    test_filenames = glob.glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack(
        [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0
    )
    test_images = test_images[:, :, :, :3]
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), config.DEVICE)

    # use normaliztation - helpful in case of segformer
    if config.USE_NORMALIZATION:

        transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        test_images = transform(test_images)

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
            p=config.p_augment,
        ),
        album.OneOf(
            [
                album.MotionBlur(p=1),
                album.MedianBlur(blur_limit=3, p=1),
                album.Blur(blur_limit=3, p=1),
            ],
            p=config.p_augment,
        ),
        album.OneOf(
            [
                album.OpticalDistortion(p=1),
                album.GridDistortion(p=1),
                album.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
            ],
            p=config.p_augment,
        ),
        album.OneOf(
            [
                album.RandomContrast(limit=0.6, p=1),
                album.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1
                ),
                album.RandomBrightness(limit=0.2, p=1),
            ],
            p=config.p_augment,
        ),
        album.RandomBrightnessContrast(p=config.p_augment),
        album.RandomGamma(p=config.p_augment),
    ]
    return album.Compose(train_transform)


def validation_augmentation():
    test_transform = [
        album.CenterCrop(
            height=config.MINHEIGHT, width=config.MINWIDTH, always_apply=True
        ),
    ]
    return album.Compose(test_transform)
