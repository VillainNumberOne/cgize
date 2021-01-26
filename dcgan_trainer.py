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

from utils import *
from dcgan_net import *

class Properties_DCGAN:
    def __init__(self):
        self.z_size = 128
        self.p_min = 2
        self.p_max = 5
        self.ch_img = 1

        self.g_ch_in = 512

        self.lr = 2e-4
        self.optim = torch.optim.Adam
        # self.criterion = nn.BCELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @property    
    def d_ch_in(self):
        return self.g_ch_in // 2 ** (self.p_max-self.p_min-1)

        print(f"Discriminator output: {Discriminator(p_min, p_max, 1, d_ch_in)(torch.randn(10, 1, 32, 32)).shape}")
        print(f"Generator output: {Generator(p_min, p_max, z_size, 1, g_ch_in)(torch.randn(10, z_size, 1, 1)).shape}")

class DCGAN:
    def __init__(self, properties, data_loader):
        self.initialize_network(properties)
        self.initialize_data(data_loader)
        

    def initialize_network(self, properties):
        self.P = Properties_DCGAN()

        self.G = Generator(self.P.p_min, self.P.p_max, self.P.z_size, self.P.ch_img, self.P.g_ch_in)
        self.D = Discriminator(self.P.p_min, self.P.p_max, self.P.ch_img, self.P.d_ch_in)

        initialize_weights(self.G)
        initialize_weights(self.D)

        self.G.to(self.P.device)
        self.D.to(self.P.device)

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.P.lr, betas=(0.5, 0.999))
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.P.lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def initialize_data(self, data_loader):
        self.DL = data_loader
        self.batch_size = self.DL.batch_size
        self.batches_number = len(self.DL)

        self.demo_z = torch.randn(6**2, self.P.z_size, 1, 1).to(self.P.device)

    def train_D(self, batch):
        D_real = self.D(batch).reshape(-1)
        D_loss_real = self.criterion(D_real, torch.ones_like(D_real))

        z = torch.randn(self.batch_size, self.P.z_size, 1, 1).to(self.P.device)
        D_fake = self.D(self.G(z)).reshape(-1)
        D_loss_fake = self.criterion(D_fake, torch.zeros_like(D_fake))

        D_loss = (D_loss_real + D_loss_fake) / 2

        self.D_opt.zero_grad()
        D_loss.backward()
        self.D_opt.step()

        return D_loss

    def train_G(self):
        z = torch.randn(self.batch_size, self.P.z_size, 1, 1).to(self.P.device)
        output = self.D(self.G(z)).reshape(-1)
        G_loss = self.criterion(output, torch.ones_like(output))

        self.G_opt.zero_grad()
        G_loss.backward()
        self.G_opt.step()

        return G_loss

    def fit(self, epochs):
        
        for epoch in range(epochs):
            for i, (images, _) in enumerate(self.DL):
                images = images.to(self.P.device)
                
                D_loss = self.train_D(images)
                G_loss = self.train_G()
                # print(i)

            # pgan_demo.refresh(epoch)

            print(f"""Epoch {epoch}:
            Discriminator loss: {D_loss}
            Generator loss: {G_loss}""")

            self.demo()

    def demo(self):
        with torch.no_grad():
            latent = self.demo_z
            images = self.G(latent).clone().detach()
            images = images.reshape(images.size(0), self.P.ch_img, 2**self.P.p_max, 2**self.P.p_max)
            
            directory = 'images'
                
            images = denorm(images)
            save_image(images, os.path.join(directory, 'demo.png'), nrow=6)


def main():
    P = Properties_DCGAN()
    DL = mnist_get_data(P.device, 100)

    dcgan = DCGAN(P, DL)

    dcgan.fit(100)

    

if __name__ == "__main__":
    main()