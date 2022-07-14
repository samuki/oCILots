from glob import glob
import re
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import config
import shutil


def make_log_dir(dt):
    os.makedirs(dt)
    shutil.copyfile("config.py", dt + "/config.py")


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return (
        np.stack(
            [np.array(Image.open(f)) for f in sorted(glob(path + "/*.png"))]
        ).astype(np.float32)
        / 255.0
    )


def load_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.array(Image.open(path)).astype(np.float32) / 255.0


def show_first_n(imgs, masks, n=5):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(5, len(imgs))
    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(masks[i])
        axs[0, i].set_title(f"Image {i}")
        axs[1, i].set_title(f"Mask {i}")
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.show()


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    else:
        return (
            torch.from_numpy(x)
            .contiguous()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )


def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)] * 3, -1))
            axs[0, i].set_title(f"Sample {i}")
            axs[1, i].set_title(f"Predicted {i}")
            axs[2, i].set_title(f"True {i}")
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
    else:  # classification
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(
                f"True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}"
            )
            axs[i].set_axis_off()
    plt.show()


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // config.PATCH_SIZE
    w_patches = y.shape[-1] // config.PATCH_SIZE
    patches_hat = (
        y_hat.reshape(
            -1, 1, h_patches, config.PATCH_SIZE, w_patches, config.PATCH_SIZE
        ).mean((-1, -3))
        > config.CUTOFF
    )
    patches = (
        y.reshape(
            -1, 1, h_patches, config.PATCH_SIZE, w_patches, config.PATCH_SIZE
        ).mean((-1, -3))
        > config.CUTOFF
    )
    return (patches == patches_hat).float().mean()


def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % config.PATCH_SIZE) + (
        w % config.PATCH_SIZE
    ) == 0  # make sure images can be patched exactly

    images = images[:, :, :, :3]

    h_patches = h // config.PATCH_SIZE
    w_patches = w // config.PATCH_SIZE
    patches = images.reshape(
        (n_images, h_patches, config.PATCH_SIZE, w_patches, config.PATCH_SIZE, -1)
    )
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, config.PATCH_SIZE, config.PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape(
        (n_images, h_patches, config.PATCH_SIZE, w_patches, config.PATCH_SIZE, -1)
    )
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > config.CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(
            np.maximum(p, np.array([l.item(), 0.0, 0.0]))
        )
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()


def create_submission(labels, test_filenames, submission_filename):
    test_path = "data/test/images"
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write(
                        "{:03d}_{}_{},{}\n".format(
                            img_number,
                            j * config.PATCH_SIZE,
                            i * config.PATCH_SIZE,
                            int(patch_array[i, j]),
                        )
                    )


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace("_", " ").title(), fontsize=20)
        if image.shape[0] != config.WIDTH:
            image = image.T
        plt.imshow(image.detach().numpy())
    plt.show()
