from glob import glob
import cv2
import numpy as np
import torch
import config
from train import train
import utils
from models.unet_new import UNet
from models.swin_transformer import SwinTransformerPretrained
from models.maskformer import MaskformerPretrained
import dataset
import datetime
from models.LRSR import LRSRModel


def main():
    # log training
    now = datetime.datetime.now()
    dt_string = now.strftime("results/%d%m%Y_%H:%M:%S")
    utils.make_log_dir(dt_string)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    if config.USE_AUGMENTATIONS:
        train_dataset = dataset.FlexibleDataset(
            "data/training",
            device,
            use_patches=False,
            resize_to=(config.HEIGHT, config.WIDTH),
            augmentation=dataset.training_augmentation(),
        )
        val_dataset = dataset.FlexibleDataset(
            "data/validation",
            device,
            use_patches=False,
            resize_to=(config.HEIGHT, config.WIDTH),
            augmentation=dataset.validation_augmentation(),
        )
    else:
        train_dataset = dataset.ImageDataset(
            "data/training", device, use_patches=False, resize_to=(384, 384)
        )
        val_dataset = dataset.ImageDataset(
            "data/validation", device, use_patches=False, resize_to=(384, 384)
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    #model = SwinTransformerPretrained().to(device)
    #model = MaskformerPretrained().to(device)
    model = LRSRModel().to(device)
    loss_fn = config.LOSS
    metric_fns = config.METRICS
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = config.EPOCHS
    train(
        train_dataloader,
        val_dataloader,
        model,
        loss_fn,
        metric_fns,
        optimizer,
        n_epochs,
        save_dir=dt_string,
    )
    
    model.load_state_dict(torch.load(dt_string+'/model.pth'))
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
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), device)
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
        test_pred, test_filenames, submission_filename=dt_string + "/submission.csv"
    )


if __name__ == "__main__":
    main()
