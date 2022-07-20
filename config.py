from models.segformer_pretrained import SegFormerPretrained
from models.unet_new import UNet
import utils
import torch
from models.LRSR import LRSRModel
from models.unet_new import UNet
#from models.swin_transformer import SwinTransformerPretrained
#from models.maskformer import MaskformerPretrained


###################################################### DATASET ######################################################


PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.2  # minimum average brightness for a mask patch to be classified as containing road
BATCH_SIZE = 4
HEIGHT = 384
WIDTH = 384
MINHEIGHT = 384
MINWIDTH = 384


# this add random crops of size HEIGHT x WIDTH and augmentations with p_augement. View dataset.__training_augmentation() for details
USE_AUGMENTATIONS = True
p_augment=0.3


###################################################### MODEL & TRAINING ######################################################


# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model
#MODEL = UNet().to(DEVICE)
#model = SwinTransformerPretrained().to(DEVICE)
#model = MaskformerPretrained().to(DEVICE)
#model = LRSRModel().to(DEVICE)
MODEL = SegFormerPretrained().to(DEVICE)

EPOCHS =5

METRICS = {"acc": utils.accuracy_fn, "patch_acc": utils.patch_accuracy_fn}
LOSS = torch.nn.BCELoss()

# optimizer 
#OPTIMIZER = torch.optim.Adam(MODEL.parameters())
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.9)

# custom learning rate schduler
use_custom_lr_scheduler= False
start_lr = 0.3
warm_up_epochs = 0

# save best model, choose option in ['val_loss', 'val_acc', 'val_patch_acc']
save_best_metric = 'val_patch_acc'
minimize_metric = False