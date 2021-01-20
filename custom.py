import numpy as np
from collections import OrderedDict
import math
import os

import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, i_c, o_c, kernel_size, stride=1, padding=0, Deconv=True, lrelu=0.2, bias=True):
        super().__init__()
        if Deconv:
            self.layer = nn.ConvTranspose2d
            self.relu = nn.LeakyReLU(lrelu)
        else:
            self.layer = nn.Conv2d
            self.relu = nn.ReLU()
            
        self.activation = nn.Sequential(
            nn.BatchNorm2d(o_c),
            self.relu
        )
        
        self.Sequential = nn.Sequential(
            self.layer(i_c, o_c, kernel_size, stride=stride, padding=padding, bias=bias),
            self.activation
        )
        
        
    def forward(self, x_b):
        return self.Sequential(x_b)
    
class ConvLayer(nn.Module):
    def __init__(self, i_c, o_c, kernel_size, stride=1, padding=0, Deconv=True, lrelu=0.2, bias=True, N=1):
        super().__init__()

        if Deconv:
            self.name = 'deconv'
        else:
            self.name = 'conv'

        self.Sequential = nn.Sequential(*[
            ConvBlock(i_c, o_c, kernel_size, stride=stride, padding=padding, Deconv=Deconv, lrelu=lrelu, bias=bias) if i==0
            else ConvBlock(o_c, o_c, kernel_size=(3,3), padding=1, Deconv=Deconv, lrelu=lrelu, bias=bias)
            for i in range(N)
        ])
        
    def forward(self, x_b):
        return self.Sequential(x_b)


class ToImage(nn.Module):
    def __init__(self, c_in, im_ch):
        super().__init__()
            
        self.Sequential = nn.Sequential(
            nn.ConvTranspose2d(c_in, im_ch, 1, 1, 0)
        )

    def forward(self, x_b):
        return self.Sequential(x_b)

class FromImage(nn.Module):
    def __init__(self, im_ch, c_in):
        super().__init__()
            
        self.Sequential = nn.Sequential(
            nn.Conv2d(im_ch, c_in, 1, 1, 0)
        )

    def forward(self, x_b):
        return self.Sequential(x_b)
    