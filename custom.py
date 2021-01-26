import numpy as np
from collections import OrderedDict
import math
import os
import copy

import torch.nn as nn
import torch



class ConvBlock(nn.Module):
    def __init__(self, i_c, o_c, kernel_size, stride=1, padding=0, Deconv=True, lrelu=0.2, bias=False):
        super().__init__()
        if Deconv:
            self.layer = nn.ConvTranspose2d
            self.relu = nn.LeakyReLU(lrelu)
            nl = 'leaky_relu'
        else:
            self.layer = nn.Conv2d
            self.relu = nn.ReLU()
            nl = 'relu'
            
        self.activation = nn.Sequential(
            self.relu,
            nn.BatchNorm2d(o_c)
        )

        #weight initialization
        self.layer = self.layer(i_c, o_c, kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.layer.weight, nonlinearity=nl)
        
        
        # self.Sequential = nn.Sequential(
        #     self.layer,
        #     self.activation
        # )

        self.scale = (torch.mean(self.layer.weight.data ** 2)) ** 0.5

    def forward(self, x):
        x = self.layer(x.mul(self.scale))
        return self.activation(x)
        
        
    # def forward(self, x_b):
    #     return self.Sequential(x_b)
    
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

class Minibatch(nn.Module):
    def __init__(self):
        super().__init__()

        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)  

    def forward(self, x_b):
        shape = list(x_b.size())
        target_shape = copy.deepcopy(shape)
        target_shape[1] = 1

        vals = self.adjusted_std(x_b, dim=0, keepdim=True)

        vals = torch.mean(vals, dim=1, keepdim=True)
        vals = vals.expand(*target_shape)

        return torch.cat([x_b, vals], 1)
    