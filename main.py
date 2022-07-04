from glob import glob
import cv2
import numpy as np
import torch
from config import CUTOFF, PATCH_SIZE
from train import train
from unet import UNet
from utils import accuracy_fn, create_submission, load_all_from_path, np_to_tensor, patch_accuracy_fn
import dataset 

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # reshape the image to simplify the handling of skip connections and maxpooling
    #train_dataset = ImageDataset('training', device, use_patches=False, resize_to=(384, 384))
    #val_dataset = ImageDataset('validation', device, use_patches=False, resize_to=(384, 384))
    
    # TODO try using two classes
    #class_names = ['background', 'street']
    class_names = ['street']
    #select_classes = ['background', 'street']
    select_classes = ['street']
    #class_rgb_values = [[0, 0, 0], [255, 255, 255]]
    class_rgb_values = [[255, 255, 255]]
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]
    train_dataset = dataset.FlexibleDataset(
        'training', 
        device,
        use_patches=False,
        resize_to=(384, 384),
        augmentation=dataset._training_augmentation(),
        preprocessing=dataset._flexible_preprocess(),
        select_class_rgb_values=select_class_rgb_values
    )

    val_dataset = dataset.FlexibleDataset(
        'validation', 
        device,
        use_patches=False,
        resize_to=(384, 384),
        augmentation=dataset._training_augmentation(), 
        preprocessing=dataset._flexible_preprocess(),
        select_class_rgb_values=select_class_rgb_values
    )
    image, mask = train_dataset[1]
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
    model = UNet().to(device)
    loss_fn = torch.nn.BCELoss()
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 35
    train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
    test_path = 'test/images'
    # predict on test set
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred= np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')
    
    
    
if __name__ == "__main__":
    main()