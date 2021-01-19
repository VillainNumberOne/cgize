import numpy as np
from collections import OrderedDict
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  Dataset
import torch.nn.functional as F

def progression(current, p_min, p_max, first=-1):
    if first == -1:
        first = 2 ** p_min
        p_min += 1

    if current == 0:
        return first
    else:
        return 2 ** min(current+p_min-1, p_max)

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(element, device) for element in data]
    else:
        return data.to(device, non_blocking=True)


class GProperties:
    def __init__(self):
        self.ch_image = 1
        self.ch_out = 16
        self.ch_in = 512
        self.latent_size = None

        self.p_min = 2
        self.p_max = 10

        self.ch_pr = [min(self.ch_out * 2 ** i, self.ch_in) for i in range(self.p_max - self.p_min)][::-1]

    def __str__(self):
        return f""" #=========== GENERATOR ===========#
        image channels: {self.ch_image}
        middle section input channels: {self.ch_in}
        middle section output channels: {self.ch_out}
        min. progress value: {self.p_min} (min. resolution = [{2**self.p_min} x {2**self.p_min}])
        max. progress value: {self.p_min} (max. resolution = [{2**self.p_max} x {2**self.p_max}])
        middle section channel progression: {self.ch_pr.__str__()}
        """

class DProperties:
    def __init__(self):
        self.ch_image = 1
        self.ch_out = 512
        self.ch_in = 16
        self.latent_size = None

        self.p_min = 2
        self.p_max = 10

        self.ch_pr = [min(self.ch_in * 2 ** i, self.ch_out) for i in range(self.p_max - self.p_min)]

    def __str__(self):
        return f""" #========= DISCRIMINATOR =========#
        image channels: {self.ch_image}
        middle section input channels: {self.ch_in}
        middle section output channels: {self.ch_out}
        min. progress value: {self.p_min} (min. resolution = [{2**self.p_min} x {2**self.p_min}])
        max. progress value: {self.p_min} (max. resolution = [{2**self.p_max} x {2**self.p_max}])
        middle section channel progression: {self.ch_pr.__str__()}
        """
    

class Properties:
    def __init__(self):
        self.G = GProperties()
        self.D = DProperties()

    def __str__(self):
        return self.G.__str__() + "\n" + self.D.__str__()

# P = Properties()
# print(P)