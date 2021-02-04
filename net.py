import numpy as np
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom import *
from utils import *

class Generator(nn.Module):
    def __init__(self, p_min, p_max, z_size, ch_img, ch_in, ch_out, p_start=-1):
        super(Generator, self).__init__()
        if p_start == -1: self.p_start = p_min
        else: self.p_start = p_start 
        self.p_inicial = int(self.p_start)

        self.p_max = p_max
        self.p_min = p_min
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_img = ch_img

        self.channels = [
            min(ch_out * 2 ** (i-self.p_min), ch_in) for i in reversed(range(p_min, p_max+1))
        ]

        # first section
        self.first_section = nn.Sequential(
            Deconv(z_size, self.channels[0], 2**p_min, 1, 0),
            Deconv(self.channels[0], self.channels[0], 3, 1, 1)
        )

        # middle section
        self.middle_section = nn.ModuleList([])
        self.middle_section.append(self.first_section)
        for i in range(self.p_min+1, self.p_start+1):
            current = i - self.p_min
            self.middle_section.append(nn.Sequential(
                Deconv(self.channels[current-1], self.channels[current], 4, 2, 1),
                Deconv(self.channels[current], self.channels[current], 3, 1, 1)
            ))

        # last section
        self.last_section = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(self.channels[self.p_start-self.p_min], self.ch_img, 1, 1, 0),
            nn.Tanh()
        )])


    def forward(self, x_b, progress=0.5):
        for index, module in enumerate(self.middle_section):
            x_b = module(x_b)
            if index == len(self.middle_section)-2 and self.p_start > self.p_inicial: 
                x_b_prev = x_b.clone().detach() # may cause problems with gradient descent
            
        x_b = self.last_section[-1](x_b)

        # fading new layers
        if self.p_start > self.p_inicial: 
            with torch.no_grad():
                x_b_prev = self.last_section[-2](x_b_prev)
                x_b_prev = nn.Upsample(2**self.p_start)(x_b_prev)

            alpha = progress - int(progress)
            x_b = x_b * alpha + x_b_prev * (1 - alpha)

        return x_b
        

    def grow(self):
        # update middle section
        old = self.p_start-self.p_min
        new = old + 1
        self.p_start += 1
        self.middle_section.append(nn.ModuleList([
                Deconv(self.channels[old], self.channels[new], 4, 2, 1),
                Deconv(self.channels[new], self.channels[new], 3, 1, 1)
            ]))
        # update last section
        self.last_section.append(
            nn.ConvTranspose2d(self.channels[new], 1, 1, 1, 0)
        )


class Discriminator(nn.Module):
    def __init__(self, p_min, p_max, ch_img, ch_in, ch_out, p_start=-1):
        super(Discriminator, self).__init__()
        if p_start == -1: self.p_start = p_min
        else: self.p_start = p_start 
        self.p_inicial = int(self.p_start)

        self.p_max = p_max
        self.p_min = p_min
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_img = ch_img

        self.channels = [
            min(ch_out * 2 ** (i-self.p_min), ch_in) for i in range(p_min, p_max+1)
        ]

        # last section
        self.last_section = nn.Sequential(
            Minibatch(),
            Conv(ch_in+1, ch_in, 3, 1, 1),
            nn.Conv2d(ch_in, ch_in, 2**p_min, 1, 0),
            nn.Flatten(),
            nn.Linear(ch_in, ch_img)
        )

        #middle section
        self.middle_section = nn.ModuleList([])
        for i in range(self.p_max-self.p_start, self.p_max-2):
            self.middle_section.append(nn.Sequential(
                Conv(self.channels[i], self.channels[i], 3, 1, 1),
                Conv(self.channels[i], self.channels[i+1], 4, 2, 1)
            ))
        self.middle_section.append(self.last_section)
        self.middle_section = self.middle_section[::-1]


        # first section 
        self.first_section = nn.ModuleList([nn.Sequential(
            Conv(self.ch_img, self.channels[self.p_max-self.p_start], 1, 1, 0),
        )])
        
        
    def forward(self, x_b):
        for module in self.first_section: x_b = module(x_b)
        for module in reversed(self.middle_section):
            x_b = module(x_b)
        return x_b

    def grow(self):
        pass


def main():
    z_size = 512
    p_min = 2
    p_max = 7
    p_start = 5

    g_ch_in = 512
    d_ch_in = g_ch_in // 2 ** (p_max-p_min-1)
    

    G = Generator(p_min, p_max, z_size, 1, g_ch_in, 16, p_start)
    D = Discriminator(p_min, p_max, 1, g_ch_in, 16, p_start)
    # print(D.first_section, D.middle_section[::-1])
    print("Generatar trainable parameters: ", sum(p.numel() for p in G.parameters() if p.requires_grad))
    print("Discriminator trainable parameters: ", sum(p.numel() for p in D.parameters() if p.requires_grad))
    print(f"Generator output: {G(torch.randn(10, z_size, 1, 1)).shape}")
    print(f"Discriminator output: {D(torch.randn(10, 1, 32, 32)).shape}")
    # G.grow()
    # print(f"Generator output: {G(torch.randn(10, z_size, 1, 1)).shape}")
    # G.grow()
    # print(f"Generator output: {G(torch.randn(10, z_size, 1, 1)).shape}")

    # print(D.channels)

    # z = torch.randn(100, 128, 1, 1)
    # D_fake = G(z).reshape(-1)
    # print(min(D_fake), max(D_fake))
    # print(G.channels)
    print("success")

if __name__ == "__main__":
    main()