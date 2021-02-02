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

from demo import *
from custom import *
from utils import *
from net import *

class Properties_PGAN:
    def __init__(self):
        self.z_size = 128
        self.p_min = 2
        self.p_max = 5
        self.ch_img = 1
        self.ch_in = 128
        self.ch_out = 16
        self.p_start = 5

        self.lr = 1e-4
        self.optim = torch.optim.Adam
        # self.criterion = nn.BCELoss()

        self.lambda_gp = 10
        self.critic_N = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @property    
    def d_ch_in(self):
        return self.g_ch_in // 2 ** (self.p_max-self.p_min-1)


class PGAN:
    def __init__(self, properties, data_loader):
        self.initialize_network(properties)
        self.initialize_data(data_loader)
        

    def initialize_network(self, properties):
        self.P = properties

        self.G = Generator(self.P.p_min, self.P.p_max, self.P.z_size, self.P.ch_img, self.P.ch_in, self.P.ch_out, self.P.p_start)
        self.D = Discriminator(self.P.p_min, self.P.p_max, self.P.ch_img, self.P.ch_in, self.P.ch_out, self.P.p_start)

        initialize_weights(self.G)
        initialize_weights(self.D)

        self.G.to(self.P.device)
        self.D.to(self.P.device)

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.P.lr, betas=(0,0.9))
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.P.lr, betas=(0,0.9))
        # self.criterion = nn.BCELoss()

        self.G.train()
        self.D.train()

    def initialize_data(self, data_loader):
        self.DL = data_loader
        self.batch_size = self.DL.batch_size
        self.batches_number = len(self.DL)

        self.demo_z = torch.randn(6**2, self.P.z_size, 1, 1).to(self.P.device)

    def train_D(self, batch):
        D_real = self.D(batch).reshape(-1)

        z = torch.randn(self.batch_size, self.P.z_size, 1, 1).to(self.P.device)
        fake = self.G(z)
        D_fake = self.D(fake).reshape(-1)

        gp = gradient_penalty(self.D, batch, fake, self.P.device)
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) + self.P.lambda_gp * gp
         
            
        self.D_opt.zero_grad()
        D_loss.backward()
        self.D_opt.step()

        return D_loss

    def train_G(self):
        z = torch.randn(self.batch_size, self.P.z_size, 1, 1).to(self.P.device)
        output = self.D(self.G(z)).reshape(-1)
        G_loss = -torch.mean(output)

        self.G_opt.zero_grad()
        G_loss.backward()
        self.G_opt.step()

        return G_loss

    def fit(self, epochs):
        
        for epoch in range(epochs):
            for i, (images, _) in enumerate(self.DL):
                images = images.to(self.P.device)

                for _ in range(self.P.critic_N):
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

    def demo_1(self):
        with torch.no_grad():
            latent = torch.randn(1, self.P.z_size, 1, 1).to(self.P.device)
            images = self.G(latent).clone().detach()
            
            directory = 'images'
                
            images = denorm(images)
            save_image(images, os.path.join(directory, 'demo_1.png'), nrow=6)


def main():
    P = Properties_PGAN()
    P.p_max = 5
    P.z_size = 64
    DL = mnist_get_data(P.device, 100, 1000)

    pgan = PGAN(P, DL)

    pgan.fit(100)
    # print("success")

    # dcgan.G.load_state_dict(torch.load('data\generator_state_dict'))
    # dcgan.G.eval()

    # dcgan.D.load_state_dict(torch.load('data\discriminator_state_dict'))
    # dcgan.D.eval()

    # dcgan.demo_1()

    # dcgan.fit(100)
            
# P = Properties()
# # P.D.lr = 5e-7
# DL = mnist_get_data(P.device, 100)

# pgan = PGAN(P, DL)
# pgan.fit(10)

if __name__ == "__main__":
    main()
