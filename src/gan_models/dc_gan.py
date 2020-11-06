"""Implement class for DC-GAN based on Pytorch framework.
"""
from __future__ import print_function
import random
import torch
import torch.nn as nn

# Size of z latent vector
nz = 100
# Size of feature maps in generator
ngf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution.
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ## More Layers ...
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution.
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ## More Layers ...
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)