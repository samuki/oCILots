import utils
import torch

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
BATCH_SIZE = 4
HEIGHT = 400
WIDTH = 400
MINHEIGHT = 384
MINWIDTH = 384

EPOCHS = 50 

METRICS = {"acc": utils.accuracy_fn, "patch_acc": utils.patch_accuracy_fn, "f1_patch_acc": utils.patch_f1_fn}
LOSS = torch.nn.MSELoss() 

# this add random crops of size HEIGHT x WIDTH and augmentations with p_augement. View dataset.__training_augmentation() for details
USE_AUGMENTATIONS = True
p_augment=0.6

# save best model, choose option in ['val_loss', 'val_acc']
save_best_metric = 'val_f1_patch_acc'
minimize_metric = False
