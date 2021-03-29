#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:46:05 2021

@author: subhadip
"""

import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cnn_data_space(nn.Module):
    def __init__(self, n_in_channels=3, n_out_channels = 1, n_filters=32, kernel_size=5):
        super(cnn_data_space, self).__init__()
        self.pad = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=1, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        
        
    def forward(self, h, y, z):
        dh = torch.cat((h, y, z), dim=1)
        dh = self.act1(self.conv1(dh))
        dh = self.act2(self.conv2(dh))
        dh = self.conv3(dh)
        return h + dh
    
class cnn_image_space(nn.Module):
    def __init__(self, n_in_channels=2, n_out_channels = 1, n_filters=32, kernel_size=5):
        super(cnn_image_space, self).__init__()
        self.pad = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=1, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        
    def forward(self, x, u):
        dx = torch.cat((x, u), dim=1)
        dx = self.act1(self.conv1(dx))
        dx = self.act2(self.conv2(dx))
        dx = self.conv3(dx)
        return x + dx
    
class LPD(nn.Module):
    def __init__(self, fwd_op, adjoint_op, niter=20, sigma=0.01, tau=0.01): 
        super(LPD, self).__init__()
        
        self.fwd_op = fwd_op
        self.adjoint_op = adjoint_op
        self.niter = niter
        self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device))
        self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device))
        self.cnn_image_layers = nn.ModuleList([cnn_image_space().to(device) for i in range(self.niter)])
        self.cnn_data_layers = nn.ModuleList([cnn_data_space().to(device) for i in range(self.niter)])
        
    def forward(self, y, x_init):
        x = x_init
        h = torch.zeros_like(y)
        for iteration in range(self.niter):
            h = self.cnn_data_layers[iteration](h, y, self.sigma[iteration] * self.fwd_op(x))
            x = self.cnn_image_layers[iteration](x, self.tau[iteration] * self.adjoint_op(h))
        return x
    
#custom weights initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, n_filters=64): #48 also kinda worked
        super(Discriminator, self).__init__()

        #A bunch of convolutions one after another
        model = [   nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(n_filters, 2*n_filters, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(2*n_filters), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(2*n_filters, 4*n_filters, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(4*n_filters), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(4*n_filters, 8*n_filters, kernel_size=4, padding=1),
                    nn.InstanceNorm2d(8*n_filters), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(8*n_filters, 1, kernel_size=4, padding=1)]
        #this last conv. layer takes in an image of size (n/16)x(n/16), when the discriminator input is of
        #size nxn. The size of the returned image is ((n/16)-1)x((n/16)-1) = 15 x 15, when n = 256
        #Therefore, the output tensor from this block is of 1x15x15.

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        out = torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return out     
    
    
    
    
    
    
    