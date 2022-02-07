import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ResNetModel as rn
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
# from torchsummary import summary

"""create model the model"""
d = 2
input_channel = 730
f = input_channel // d
n_classes = 10
stride_first_layer = 2
ResNetModel = rn.ResNet(input_channel, f, stride_first_layer, n_classes)

x = torch.ones((730, 300, 600))
x2 = ResNetModel(x[None, ...])