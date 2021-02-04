import numpy as np
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom import *
from utils import *

def last_section_G(ch_in, ch_out=1):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
        nn.Tanh()
    )

def layers_G(ch_in, ch_out):
    return nn.Sequential(
        Deconv(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
        Deconv(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
    )

class Generator(nn.Module):
    def __init__(self, p_min, p_max, z_size, ch_img, ch_in, ch_out, p_start):
        super(Generator, self).__init__()

        self.p_max = p_max
        self.p_min = p_min
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.ch_img = ch_img

        self.p_start = p_start
        self.p_current = int(p_start)

        self.channels = [min(ch_out * 2 ** (i-self.p_min), ch_in) for i in reversed(range(p_min, p_max+1))]

        self.model =                Generator_base(self.channels[:p_start-p_min+1])
        self.old_last_section =     None
        self.new_last_section =     last_section_G(self.channels[p_start-p_min])
        self.old_layers =           None
        self.new_layers =           None

        self.model =                Generator_model(self.model, self.new_last_section)

    def grow(self):
        assert self.p_current+1 <= self.p_max, f"current progress value ({self.p_current+1}) exceeds maximum ({self.p_max})"
        self.p_current += 1

        if self.new_layers: self.model = nn.Sequential(self.model, self.new_layers)
            
        self.new_layers =           layers_G(self.channels[self.p_current-self.p_min-1], self.channels[self.p_current-self.p_min])
        self.old_last_section =     self.new_last_section
        self.new_last_section =     last_section_G(self.channels[self.p_current-self.p_min])

        self.model = Generator_model(
            self.model, 
            new_last_section = self.new_last_section, 
            old_last_section = self.old_last_section, 
            new_layers = self.new_layers
        )

class Generator_model(nn.Module):
    def __init__(self, base_model, new_last_section, new_layers=None, old_last_section=None):
        super(Generator_model, self).__init__()

        self.base_model = base_model
        self.old_last_section = old_last_section
        self.new_last_section = new_last_section
        self.new_layers = new_layers

    def forward(self, x_b, progress=None, drop_last_section=True):
        if progress:
            x_b = self.base_model(x_b)
            with torch.no_grad():
                x_b_old = self.old_last_section(x_b)
                x_b_old = nn.Upsample(scale_factor=2)(x_b_old)

            x_b = self.new_layers(x_b)
            x_b = self.new_last_section(x_b)
            
            alpha = progress - int(progress)
            x_b = x_b * alpha + x_b_old * (1 - alpha)

        else: 
            x_b = self.base_model(x_b)
            if self.new_layers: x_b = self.new_layers(x_b)
            if not drop_last_section: x_b = self.new_last_section(x_b)

        return x_b


class Generator_base(nn.Module):
    def __init__(self, channels, p_min=2):   
        super(Generator_base, self).__init__()
        self.layers = []

        #first section
        self.layers.append(
            Deconv(channels[0], channels[0], 2**p_min, 1, 0)
        )

        #middle section
        for i in range(len(channels)-1):
            self.layers.extend([
                Deconv(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                Deconv(channels[i+1], channels[i+1], kernel_size=3, stride=1, padding=1)
            ])

        self.Sequential = nn.Sequential(*self.layers)

    def forward(self, x_b):
        return self.Sequential(x_b)

class Discriminator_base(nn.Module):
    def __init__(self, channels, ch_img=1, p_min=2):
        super(Discriminator_base, self).__init__()
        self.layers = []

        #middle section
        for i in range(len(channels)-1):
            self.layers.extend([
                Conv(channels[i], channels[i], kernel_size=4, stride=2, padding=1),
                Conv(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1)
            ])

        # last section
        self.layers.extend([
            Conv(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=2**p_min, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(channels[-1], ch_img)
        ])

        self.Sequential = nn.Sequential(*self.layers)
            
    
    def forward(self, x_b):
        return self.Sequential(x_b)


def main():
    # 32 16 8 4
    channels = [128, 64, 32]
    z_size = channels[0]

    z_size = 128
    p_min = 2
    p_max = 5
    p_start = 5

    g_ch_in = 128
    d_ch_in = g_ch_in // 2 ** (p_max-p_min-1)
    


    # torch.manual_seed(10)
    # np.random.seed(0)

    zg = torch.randn(10, z_size, 1, 1)
    zd = torch.randn(10, 16, 2**p_start, 2**p_start)

    D = Discriminator_base([16, 32, 64, 128])

    print(f"Generator output: {D(zd).shape}")

    score = []

    # for i in range(20):
    #     G = Generator(p_min, p_max, z_size, 1, g_ch_in, 16, p_start)
    #     opt = torch.optim.Adam(G.model.parameters(), lr=2e-4, betas=(0,0.9))
    #     # print(f"Generator output: {G.model(torch.randn(10, z_size, 1, 1), drop_last_section=False).shape}")
    #     for _ in range(20):
    #         loss = torch.mean(G.model(zg, drop_last_section=False))
    #         # print(loss)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     G.grow()
    #     # print(f"Generator output")

    #     opt = torch.optim.Adam(G.model.parameters(), lr=2e-4, betas=(0,0.9))
    #     for _ in range(30):
    #         loss = torch.mean(G.model(zg, drop_last_section=False, progress=0.5))
    #         # print(loss)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     score.append(loss.item())
    #     print(i, loss.item())
    
    # print(np.mean(score))

    print("success")

if __name__ == "__main__":
    main()