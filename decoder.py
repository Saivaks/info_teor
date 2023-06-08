#!/usr/bin/env python
# coding: utf-8

# In[206]:


import pandas as pd
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
import random
import cv2
from PIL import Image
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights
from torchvision import models
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import configparser
from ast import literal_eval
import base64
import re
import hafman
import torchvision


# In[207]:


#def dekvantovanie(batch, level):
#    matrix = np.fromfunction(lambda b, i, j, k: (k+1)/(k+1)*(1/(2**level)), (batch.shape[0], 512, 16, 16), dtype=float)
#    return (batch - 0.5)/(torch.from_numpy(matrix).to('cuda'))


# In[208]:


def dekvantovanie(batch, level):
    return (batch - 0.5)/(2**level)


# In[209]:


class Decoder(nn.Module):
    def __init__(self,in_channels=512):
        super(Decoder, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
#                                    nn.BatchNorm2d(512),
#                                    nn.ReLU(inplace=True))
        
        self.upconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=256, kernel_size=2,stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(256),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(256, 256, kernel_size=3,padding=1),
#                                    nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(128),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(128, 128, kernel_size=3,padding=1),
#                                    nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(64),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(64, 64, kernel_size=3,padding=1),
#                                    nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2,stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.upconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2,stride=2)
        self.conv6 = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(3),
                                    nn.Sigmoid())
        
        
    def forward(self, x, level):
        x = x.reshape(1,512,16,16)
        x = dekvantovanie(x, level)
        x = x.float()
        x = self.upconv1(x)
        x = self.conv2(x)+x
        x = self.upconv2(x)
        x = self.conv3(x)+x
        x = self.upconv3(x)
        x = self.conv4(x)+x
        x = self.upconv4(x)
        x = self.conv5(x)+x
        x = self.upconv5(x)
        x = self.conv6(x)
        return x


# In[210]:


config = configparser.ConfigParser()
config.read('config_decoder.ini', encoding="utf-8")


# In[211]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path_load = config['INFO']['path_comp']
level = int(config['INFO']['b'])
path_save = config['INFO']['path_save']
path_weight =  config['INFO']['path_weight']


# In[212]:


model = Decoder().to(device)
model.load_state_dict(torch.load(path_weight))


# In[213]:


path_load


# In[214]:


def read_image(path, level):
    with open(path, 'rb') as file:
        text = file.read()
    
    text_list = hafman.decod(text)
    text = ''.join(text_list)
    text_list = text.split()
    text_list = [float(i) for i in text_list]
    tens = torch.FloatTensor(text_list).to(device)
    out = model(tens, level)
    return out


# In[215]:


def save_image(image, path):
    torchvision.utils.save_image(image, path)


# In[216]:


image = read_image(path_load, level)
save_image(image, path_save)


# In[ ]:





# In[ ]:




