<<<<<<< HEAD

from glob import glob
import cv2
import numpy as np
=======
>>>>>>> origin/main
import torch

import config
from train import train
import utils
<<<<<<< HEAD

from models.unet_new import UNet

from models.segformer_pretrained import SegFormerPretrained

import dataset
import datetime
import torchvision
=======
import dataset
import datetime
import test
import numpy as np
import random
>>>>>>> origin/main


def main():
    # log training
    now = datetime.datetime.now()
    dt_string = now.strftime("results/%d%m%Y_%H:%M:%S")
    utils.make_log_dir(dt_string)
    device = config.DEVICE

    # set random seed wherever possible
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # reshape the image to simplify the handling of skip connections and maxpooling
    
    """
    feature_extractor = SegformerFeatureExtractor(reduce_labels=False, size=400)
    
    train_dataset = datasetSegmentation.SemanticSegmentationDataset(
        feature_extractor=feature_extractor, device=device
    )

    val_dataset = datasetSegmentation.SemanticSegmentationDataset(
        feature_extractor=feature_extractor, train=False, device=device
    )
    """
    
    if config.USE_AUGMENTATIONS:
        train_dataset = dataset.FlexibleDataset(
<<<<<<< HEAD
            "data/trainingBigBig",
=======
            config.TRAIN_PATH,
>>>>>>> origin/main
            device,
            use_patches=False,
            resize_to=(config.HEIGHT, config.WIDTH),
            augmentation=dataset.training_augmentation(),
        )
        val_dataset = dataset.FlexibleDataset(
            config.VAL_PATH,
            device,
            use_patches=False,
            resize_to=(config.HEIGHT, config.WIDTH),
            augmentation=dataset.validation_augmentation(),
        )
    else:
        train_dataset = dataset.ImageDataset(
            config.TRAIN_PATH, device, use_patches=False, resize_to=(384, 384)
        )
        val_dataset = dataset.ImageDataset(
            config.VAL_PATH, device, use_patches=False, resize_to=(384, 384)
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
<<<<<<< HEAD
    # model = UNet().to(device)
    
    model = SegFormerPretrained().to(device)
    
    loss_fn = config.LOSS
    metric_fns = config.METRICS
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
=======
    model = config.MODEL
    loss_fn = config.LOSS
    metric_fns = config.METRICS
    optimizer = config.OPTIMIZER
>>>>>>> origin/main
    n_epochs = config.EPOCHS

    if config.LOAD_CKTP:
        model.load_state_dict(torch.load(config.CKPT_PATH+'/model.pth'))
        model.eval()


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
    
    # load best model
    model.load_state_dict(torch.load(dt_string+'/model.pth'))
    model.eval()
<<<<<<< HEAD
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


    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

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
=======

    # load test dataset
    test_images, size, test_filenames = dataset.load_test_data()

    # create test predictions
    test_pred = test.test_prediction(model, test_images, size, cutoff = config.CUTOFF)
    
    # create submision file
>>>>>>> origin/main
    utils.create_submission(
        test_pred, test_filenames, submission_filename=dt_string + "/submission.csv"
    )


if __name__ == "__main__":
    main()

