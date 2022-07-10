import utils
import torch

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
BATCH_SIZE = 4
HEIGHT = 384
WIDTH =  384
MINHEIGHT=384
MINWIDTH=384

EPOCHS =10

METRICS = {'acc': utils.accuracy_fn, 'patch_acc': utils.patch_accuracy_fn}
LOSS = torch.nn.BCELoss()

USE_AUGMENTATIONS = False
