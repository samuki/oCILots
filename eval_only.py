from glob import glob
import cv2
import numpy as np
import torch
import config
from train import train
import utils
from models.segformer_pretrained import SegFormerPretrained

import dataset
import datetime
import torchvision


def main():
    # log training
    now = datetime.datetime.now()
    model_path = './results/24072022_18:35:49'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SegFormerPretrained().to(device)

    model.load_state_dict(torch.load(model_path+'/model.pth'))
    model.eval()
    test_path = "data/test/images"
    # predict on test set
    test_filenames = glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack(
        [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0
    )
    test_images = test_images[:, :, :, :3]

    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    test_images = transform(test_images)

    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack(
        [cv2.resize(img, dsize=size) for img in test_pred], 0
    )  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape(
        (
            -1,
            size[0] // config.PATCH_SIZE,
            config.PATCH_SIZE,
            size[0] // config.PATCH_SIZE,
            config.PATCH_SIZE,
        )
    )
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > config.CUTOFF)
    utils.create_submission(
        test_pred, test_filenames, submission_filename=model_path + "/submission2.csv"
    )


if __name__ == "__main__":
    main()
