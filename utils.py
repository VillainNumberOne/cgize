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

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

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

def downsample(batch, progress=None):
    if progress:
        factor = batch.size()[3] // 2 ** progress
        if factor == 1: return batch
        else: return nn.AvgPool2d(factor)(batch)
    else: return batch

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def gradient_penalty(D, real, fake, device, mode='fit', p=None):
    batch_size, ch, h, w = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, ch, h, w).to(device)
    interpolated = real * alpha + fake * (1 - alpha)

    if mode == 'fit': mixed_scores = D(interpolated, no_fs=False)
    if mode == 'stab': mixed_scores = D(interpolated, p)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    

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
                transform=Compose([Pad(2), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

    mnist, _ = torch.utils.data.random_split(mnist, [N, len(mnist)-N])  
    data_loader = DataLoader(mnist, batch_size, shuffle=True, drop_last=True)

    return data_loader

def dem_get_data(path, batch_size, N=1000):
    ds = torchvision.datasets.ImageFolder(root=path,
    transform=Compose([
        tt.Resize((128,128)), 
        tt.Grayscale(),
        tt.RandomHorizontalFlip(),
        tt.RandomVerticalFlip(),
        ToTensor(), 
        Normalize(mean=(0.5,), std=(0.5,))
    ]))

    ds, _ = torch.utils.data.random_split(ds, [N, len(ds)-N])  

    return DataLoader(ds, batch_size, shuffle=True, drop_last=True)


def main():
    z = torch.randn(10, 1, 32, 32)
    # print(upsample(z, 4).size())
    print(downsample(z, 2).size())

if __name__ == "__main__":
    main()
