import torch
import argparse 
import yaml
from models.segformer_pretrained import SegFormerPretrained
from models.unet_new import UNet
import utils
import torch
from models.LRSR import LRSRModel
from models.maskformer import MaskformerPretrained

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        config = dotdict(yaml.safe_load(ymlfile))
        config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        config.MODEL = eval(config.MODEL)
        config.LOSS = eval(config.LOSS)
        config.OPTIMIZER = eval(config.OPTIMIZER)
    return config


parser = argparse.ArgumentParser()
input_group = parser.add_argument_group('input_group')
input_group.add_argument('--config', dest='config', required=True, type=str)
args = parser.parse_args()
config =load_config(args.config)