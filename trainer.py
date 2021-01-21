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

class PGAN:
    def __init__(self, properties, data_loader):
        self.initialize_network(properties)
        self.initialize_data(data_loader)
        

    def initialize_network(self, properties):
        self.P = properties

        self.G = Generator(self.P.G)
        self.D = Discriminator(self.P.D)

        self.G.to(self.P.device)
        self.D.to(self.P.device)

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.P.G.lr)
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.P.D.lr)

    def initialize_data(self, data_loader):
        self.DL = data_loader
        self.batch_size = self.DL.batch_size
        self.batches_number = len(self.DL)

    def reset_grad(self):
        self.G_opt.zero_grad()
        self.D_opt.zero_grad()

    def train_D(self, batch):
        real_labels = torch.ones(len(batch), 1).to(self.P.device)
        fake_labels = torch.zeros(len(batch), 1).to(self.P.device)

        # Loss for real images
        outputs = self.D(batch)
        D_loss_real = self.P.loss(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(len(batch), self.P.latent, 1, 1).to(self.P.device)
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        D_loss_fake = self.P.loss(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        D_loss = D_loss_real + D_loss_fake
        self.reset_grad()
        D_loss.backward()
        self.D_opt.step()

        return D_loss, real_score, fake_score

    def train_G(self):
        # Generate fake images and calculate loss
        z = torch.randn(self.batch_size, self.P.latent, 1, 1).to(self.P.device)
        fake_images = self.G(z)

        labels = torch.ones(self.batch_size, 1).to(self.P.device)
        G_loss = self.P.loss(self.D(fake_images), labels)

        # Backprop and optimize
        self.reset_grad()
        G_loss.backward()
        self.G_opt.step()

        return G_loss, fake_images

    def fit(self, epochs):
        d_losses, g_losses, real_scores, fake_scores = [], [], [], []

        for epoch in range(epochs):
            for i, (images, _) in enumerate(self.DL):
                images = images.to(self.P.device)
                
                D_loss, real_score, fake_score = self.train_D(images)
                G_loss, fake_images = self.train_G()
                print(i)

            # pgan_demo.refresh(epoch)

            print(f"""Epoch {epoch}:
            Discriminator loss: {D_loss}; Real score: {real_score.mean().item()}; Fake score{fake_score.mean().item()};
            Generator loss: {G_loss}""")

            self.demo()

    def demo(self, size=6):
        with torch.no_grad():
            latent = torch.randn(size**2, self.P.G.ch_in, 1, 1).to(self.P.device)
            images = self.G(latent).clone().detach()
            images = images.reshape(images.size(0), self.P.G.ch_image, 2**self.P.G.p_max, 2**self.P.G.p_max)
            
            directory = 'images'
                
            images = denorm(images)
            save_image(images, os.path.join(directory, 'demo.png'), nrow=size)
            
P = Properties()
# P.D.lr = 5e-7
DL = mnist_get_data(P.device, 100)

pgan = PGAN(P, DL)
pgan.fit(10)

