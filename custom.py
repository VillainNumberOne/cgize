import numpy as np
from collections import OrderedDict
import math
import os
import copy

import torch.nn as nn
import torch

class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.Sequential = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(ch_out, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x_b):
        return self.Sequential(x_b)

class Deconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Deconv, self).__init__()
        self.Sequential = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

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

#rewrite
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
    