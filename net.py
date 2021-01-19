import numpy as np
from collections import OrderedDict
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  Dataset
import torch.nn.functional as F

import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, Pad
from torchvision.datasets import MNIST
import torchvision.transforms as tt
from torchvision.utils import save_image
from torchvision.datasets.utils import download_url

from blocks import *
from utils import *

class Generator(nn.Module):
    def __init__(self, properties):
        super(Generator, self).__init__()
        # initialize parameters
        self.P = GProperties()

        # initialize network
        self.layers = []
        # first section
        self.layers.extend([ 
            ConvBlock(self.P.ch_in, self.P.ch_in, 2**self.P.p_min, 1, 0), 
            ConvBlock(self.P.ch_in, self.P.ch_in, 3, 1, 1)
        ])

        # middle section 
        for i in range(len(self.P.ch_pr)):
            self.layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBlock(self.P.ch_in, self.P.ch_pr[i], 3, 1, 1) if i == 0 else ConvBlock(self.P.ch_pr[i-1], self.P.ch_pr[i], 3, 1, 1), 
                ConvBlock(self.P.ch_pr[i], self.P.ch_pr[i], 3, 1, 1) 
            ])

        # last section 
        self.layers.append(ToImage(self.P.ch_pr[-1], self.P.ch_image))

        self.Sequential = nn.Sequential(*self.layers)



    def forward(self, x_b):
        # standard, exept the case when the P value changes (then grow)
        return self.Sequential(x_b)

    def grow(self):
        # create a new model with identical modules, copy state_dict of each old module to the new model
        # delete the old model
        # add new layers and initialize them
        pass




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator(None)
to_device(G, device)
t = torch.randn(1, 512, 1, 1).to(device)

print(G(t).shape)


