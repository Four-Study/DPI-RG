#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(17)


# In[5]:


transform = transforms.Compose([ 
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder('~/Downloads/tiny-imagenet-200/train', transform=transform)


# In[6]:


train_data, val_data, test_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])


# In[7]:


batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)


# In[8]:


from torchvision.utils import make_grid

for images, _ in train_loader:
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
    break


# In[ ]:




