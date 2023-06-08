#!/usr/bin/env python
# coding: utf-8

# In[100]:


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
import torchvision
import hafman


# In[101]:


def kvantovanie(batch, level):
    matrix = np.fromfunction(lambda b, i, j, k: (k+1)/(k+1)*(2**level), (batch.shape[0], 512, 16, 16), dtype=float)
    return batch*torch.from_numpy(matrix).to('cuda') + 0.5


# In[102]:


class Coder(nn.Module):
    def __init__(self,in_channels=512):
        super(Coder, self).__init__()
        self.coder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.coder = nn.Sequential(*list(self.coder.children())[:-2])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, level):
        embed = self.coder(x)
        out = self.sigmoid(embed)
        out = kvantovanie(out, level)
        out = out.flatten(1)
        return out


# In[103]:


config = configparser.ConfigParser()
config.read('config_coder.ini', encoding="utf-8")


# In[104]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path_load = config['INFO']['path_image']
level = int(config['INFO']['b'])
path_save = config['INFO']['path_save']
path_weight =  config['INFO']['path_weight']


# In[105]:


def read_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = np.array(img)
    img = img/255
    img = transforms.ToTensor()(img).float()
    img = img.reshape(1,3,512,512)
    return img


# In[106]:


tenzor_img = read_image(path_load).to(device)
model = Coder().to(device)


# In[107]:


model.load_state_dict(torch.load(path_weight))


# In[108]:


def coding(model, level, path_save, image):
    data = model(image, level)
    
    arr = ' '.join([str(int(item)) for item in data[0].detach().cpu().numpy().astype(int)])
    haf = hafman.cod(arr)
    text_file = open(path_save, "wb")
    text_file.write(haf)
    text_file.close()


# In[109]:


coding(model, level, path_save, tenzor_img)


# In[ ]:





# In[ ]:




