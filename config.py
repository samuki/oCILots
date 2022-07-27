from models.segformer_pretrained import SegFormerPretrained
from models.unet_new import UNet
import utils
import torch
from models.LRSR import LRSRModel
from models.unet_elu_crf import UNet
#from models.swin_transformer import SwinTransformerPretrained
#from models.maskformer import MaskformerPretrained


###################################################### DATASET ######################################################

TRAIN_PATH = 'data/training'
VAL_PATH = 'data/validation'
TEST_PATH = 'data/test/images'


PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road
BATCH_SIZE = 4
HEIGHT = 400
WIDTH = 400
MINHEIGHT = 384
MINWIDTH = 384

# gradient accumulation which allows for larger batch sizes
# the effective batch size is BATCH_SIZE*GRAD_ACCUM
GRAD_ACCUM = 1

# this add random crops of size HEIGHT x WIDTH and augmentations with p_augement. View dataset.__training_augmentation() for details
USE_AUGMENTATIONS = True
p_augment=0


###################################################### MODEL & TRAINING ######################################################

RANDOM_SEED=42

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model
MODEL = UNet().to(DEVICE)
#MODEL = SwinTransformerPretrained().to(DEVICE)
#MODEL = MaskformerPretrained().to(DEVICE)
#model = LRSRModel().to(DEVICE)
#MODEL = SegFormerPretrained().to(DEVICE)

# load ckpt to fine-tune
LOAD_CKTP=False
CKPT_PATH="results/25072022_13:41:25"

EPOCHS=10

METRICS = {"acc": utils.accuracy_fn, "patch_acc": utils.patch_accuracy_fn, "f1_patch_acc": utils.patch_f1_fn}
LOSS = torch.nn.MSELoss() 

# optimizer 
OPTIMIZER=torch.optim.Adam(MODEL.parameters())
#OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.9)

# custom learning rate scheduler
use_custom_lr_scheduler=False
start_lr=0.3
warm_up_epochs=0

# save best model, choose option in ['val_loss', 'val_acc', 'val_patch_acc']
save_best_metric = 'val_f1_patch_acc'
minimize_metric = False
