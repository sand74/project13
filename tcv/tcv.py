import torch.nn as nn

from utils import *


def get_weight(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting weights in format layer1.layer2. ... .layerN.
       Returns:
         Return weight matrix."""
    layer = get_layer(model, path)
    return layer.state_dict()['weight']


def get_bias(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting bias in format layer1.layer2. ... .layerN.
       Returns:
         Return bias matrix"""
    layer = get_layer(model, path)
    return layer.state_dict()['bias']
