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

def upsample(batch, factor):
    return nn.Upsample(scale_factor = factor)(batch)

def downsample(batch, factor):
    return nn.AvgPool2d(factor)(batch)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.DataLoader = dl
        self.device = device
        self.batch_size = dl.batch_size
        
    def __iter__(self):
        for batch in self.DataLoader:
            yield to_device(batch, self.device)
            
    def __len__(self):
        return len(self.DataLoader)

def mnist_get_data(device, batch_size, N=1000):
    mnist = MNIST(root='data', 
                train=True, 
                download=True,
                transform=Compose([Pad(2), ToTensor(), Normalize(mean=(0.5,), std=(0.5,)), AddGaussianNoise(0., 1.)]))

    mnist, _ = torch.utils.data.random_split(mnist, [N, len(mnist)-N])  
    data_loader = DeviceDataLoader(DataLoader(mnist, batch_size, shuffle=True, drop_last=True), device)

    return data_loader


class GProperties:
    def __init__(self):
        self.ch_image = 1
        self.ch_out = 16
        self.ch_in = 512
        self.latent_size = None

        self.p_min = 2
        self.p_max = 5

        self.lr = 1e-5

    @property
    def ch_pr(self):
        return [min(self.ch_out * 2 ** i, self.ch_in) for i in range(self.p_max - self.p_min)][::-1]

    def __str__(self):
        return f""" #=========== GENERATOR ===========#
        image channels: {self.ch_image}
        middle section input channels: {self.ch_in}
        middle section output channels: {self.ch_out}
        min. progress value: {self.p_min} (min. resolution = [{2**self.p_min} x {2**self.p_min}])
        max. progress value: {self.p_min} (max. resolution = [{2**self.p_max} x {2**self.p_max}])
        middle section channel progression: {self.ch_pr.__str__()}
        learning rate: {self.lr}
        """

class DProperties:
    def __init__(self):
        self.ch_image = 1
        self.ch_out = 512
        self.ch_in = 16
        self.latent_size = None

        self.p_min = 2
        self.p_max = 5

        self.lr = 1e-5

    @property
    def ch_pr(self):
        return [min(self.ch_in * 2 ** i, self.ch_out) for i in range(self.p_max - self.p_min)]

    def __str__(self):
        return f""" #========= DISCRIMINATOR =========#
        image channels: {self.ch_image}
        middle section input channels: {self.ch_in}
        middle section output channels: {self.ch_out}
        min. progress value: {self.p_min} (min. resolution = [{2**self.p_min} x {2**self.p_min}])
        max. progress value: {self.p_min} (max. resolution = [{2**self.p_max} x {2**self.p_max}])
        middle section channel progression: {self.ch_pr.__str__()}
        learning rate: {self.lr}
        """
    

class Properties:
    def __init__(self): 
        self.G = GProperties()
        self.D = DProperties()

        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.latent = self.D.ch_out

    def __str__(self):
        return self.G.__str__() + "\n" + self.D.__str__()

# P = Properties()
# print(P)

dl = mnist_get_data(torch.device('cpu'), 10)
# print(len(dl), "\n", dl.batch_size)
# for batch in dl:
#     # print(batch.shape)
#     print(batch)
#     for image, _ in batch:
#         print(max(image), min(image))
#         break
#     break
