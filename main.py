import torch
import config
from train import train
import utils
import dataset
import datetime
import test
import numpy as np
import random


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
    if config.USE_AUGMENTATIONS:
        train_dataset = dataset.FlexibleDataset(
            config.TRAIN_PATH,
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
    model = config.MODEL
    loss_fn = config.LOSS
    metric_fns = config.METRICS
    optimizer = config.OPTIMIZER
    n_epochs = config.EPOCHS

    if config.LOAD_CKTP:
        print(f"loaded model from {config.CKPT_PATH + '/model.pth'}")
        model.load_state_dict(torch.load(config.CKPT_PATH + "/model.pth"))
        model.eval()

    #  train(
    #      train_dataloader,
    #      val_dataloader,
    #      model,
    #      loss_fn,
    #      metric_fns,
    #      optimizer,
    #      n_epochs,
    #      save_dir=dt_string,
    #  )

    import os
    base_dir = "results/christoffer"
    del train_dataloader, val_dataloader
    for dset, dname in [(train_dataset, "training"), (val_dataset, "validation")]:
        dloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)
        os.makedirs(f"{base_dir}/{dname}")
        for i, (im, gt) in enumerate(dloader):
            pred = model(im).detach().squeeze().cpu().numpy()
            gt = gt.squeeze().cpu().numpy()
            np.save(f"{base_dir}/{dname}/pred_{i}.npy", pred)
            np.save(f"{base_dir}/{dname}/gt_{i}.npy", gt)
    del dset, dloader, train_dataset, val_dataset

    # load best model
    #  model.load_state_dict(torch.load(dt_string + "/model.pth"))
    #  model.eval()

    # load test dataset
    test_images, size, test_filenames = dataset.load_test_data()
    import re
    os.makedirs(f"{base_dir}/test")
    for im, fname in zip(test_images, test_filenames):
        nr = int(re.search("\d+", fname)[0])
        pred = model(im.unsqueeze(0)).detach().squeeze().cpu().numpy()
        np.save(f"{base_dir}/test/pred_{nr}.npy", pred)

    # create test predictions
    test_pred, probabs = test.test_prediction(
        model, test_images, size, cutoff=config.CUTOFF
    )
    np.save(dt_string + "/predictions", probabs)
    # create submision file
    utils.create_submission(
        test_pred, test_filenames, submission_filename=dt_string + "/submission.csv"
    )


if __name__ == "__main__":
    main()
