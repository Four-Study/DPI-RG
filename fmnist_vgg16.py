#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
import torch.optim as optim

import numpy as np

from models import VGG16

# hyper-parameters
device         = torch.device("cuda")
ngpu           = torch.cuda.device_count()
nc             = 1
nclass         = 10
batch_size     = 256
lr             = 0.01
lambda_lr      = lambda epoch: 0.5 ** (epoch // 20)
momentum       = 0.9
weight_decay   = 5e-4
n_epoch        = 30
loss_criterion = nn.CrossEntropyLoss()

# preparing data set
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
A    = torchvision.datasets.FashionMNIST('./datasets', train=True, download=True,transform=transform)
test = torchvision.datasets.FashionMNIST('./datasets', train=False, download=True,transform=transform)


# In[2]:


idxs = np.array([])
for i in range(10):
    temp = np.where(A.targets == i)[0]
    idxs = np.concatenate((idxs, np.random.choice(temp, int(len(A) / nclass / 2), replace = False)))
idxs_ = np.setdiff1d(np.arange(len(A)), idxs)
idxs = torch.Tensor(idxs).int()
idxs_ = torch.Tensor(idxs_).int()

# np.randoam.permutation(10)
train    = torch.utils.data.Subset(A, idxs)
cal      = torch.utils.data.Subset(A, idxs_)
# trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
trainset = torch.utils.data.DataLoader(A, batch_size=batch_size, shuffle=True)
calibset = torch.utils.data.DataLoader(cal, batch_size=batch_size, shuffle=False)
testset  = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


# In[3]:


# Neural network structure
Net          = VGG16(ngpu, nc, nclass)
# Net.conv1    = nn.Conv2d(nc, Net.conv1.weight.shape[0], 3, 1, 1, bias=False) 
# Net.maxpool  = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
Net          = Net.to(device)
# Net          = nn.DataParallel(Net)
optimizer    = torch.optim.Adam(Net.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer    = optim.SGD(Net.parameters(),
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
lambda_lr    = lambda epoch: 0.5 ** (epoch // 20)
# lambda_lr    = lambda epoch: 0.85 ** epoch
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=lambda_lr)


# In[4]:


for epoch in range(n_epoch): 
    for data in trainset:  
        X, y = data
        X = X.to(device)
        y = y.to(device) 
        Net.zero_grad()  
        output = Net(X)  
        loss = loss_criterion(output, y)  

        # Backpropergation 
        loss.backward()  
        optimizer.step()  
    lr_scheduler.step()
    print('epoch', epoch, ':', loss.item())


# In[5]:


# predict accuracy on training data set
# record the sorted scores, associate permutation indexes, and ground-truth y
correct = 0
total = 0
with torch.no_grad():
    for data in trainset:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        output = Net(X)
        probs = F.softmax(output, 1)
        sorted_probs, indexes = torch.sort(probs, dim = 1, descending = True)

        for idx, feature in enumerate(output):
            if torch.argmax(feature) == y[idx]:
                correct += 1
            total += 1
acc_insample = correct / total
print('In-sample Accuracy', acc_insample)


# In[6]:


# predict accuracy on test data set
# record the sorted scores, associate permutation indexes, and ground-truth y
correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        output = Net(X)
        probs = F.softmax(output, 1)
        sorted_probs, indexes = torch.sort(probs, dim = 1, descending = True)

        for idx, feature in enumerate(output):
            if torch.argmax(feature) == y[idx]:
                correct += 1
            total += 1
acc_outsample = correct / total
print('Out-of-sample Accuracy', acc_outsample)

