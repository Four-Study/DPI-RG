#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
tik = datetime.datetime.now()

import sys
sys.path.insert(1, '../')

import itertools

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models


from models_cifar10 import GeneratorA2B
from models_cifar10 import GeneratorB2A
from models_cifar10 import DiscriminatorA

from utils import train_al
from utils import get_p_and_fake_C
from utils import visualize_p
from utils import visualize_fake_C
from utils import weights_init_normal
from utils import weights_init
from utils import LambdaLR

# import different loss functions for GAN B
from geomloss import SamplesLoss

import os
import numpy as np


# In[3]:


chi                 = True
start_epoch         = 1
n_epochs1           = 120
n_epochs2           = 151
n_rep               = 2
batch_size          = 256
lr                  = 0.002
# lams                = [2.0, 0.5, 2.0, 1.0, 0.2] # coefficients for all losses
lams                = [5.0, 1.0, 5.0, 5.0, 0.6] # coefficients for all losses
# decay_epoch         = n_epochs//2
decay_epoch2        = n_epochs2//2
ngf                 = 48
ndf                 = 32
im_size             = 64
nz                  = 4
nc                  = 3
ngpu                = torch.cuda.device_count()
cuda                = torch.cuda.is_available()
device              = torch.device("cuda" if cuda else "cpu")
display             = 300

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Resize(im_size),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset_A    = torchvision.datasets.CIFAR10(root="../datasets",train=True, transform=transform, download=True)
testset_A     = torchvision.datasets.CIFAR10(root="../datasets",train=False, transform=transform, download=True)
missing_label = [7]
present_label = [x for x in range(10) if x != 7]
all_label     = present_label + missing_label 
classes       = trainset_A.classes
idxs          = torch.where(torch.Tensor([x in present_label for x in trainset_A.targets]))[0] 
present_idxs_ = np.where(np.array([x in present_label for x in testset_A.targets]))[0] 
missing_idxs_ = np.where(np.array([x in missing_label for x in testset_A.targets]))[0] 
train_data    = torch.utils.data.Subset(trainset_A, idxs)
# test_data     = torch.utils.data.Subset(testset_A, idxs_)

# train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader   = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# In[4]:


# logger
class graphs:
    def __init__(self):
        self.cur_epochs      = []
        self.loss_G_GAN      = []
        self.loss_GAN_A2B    = []
        self.loss_GAN_B2A    = []
        self.loss_D_A        = []
        self.loss_G          = []
        self.loss_G_cycle    = []
        

# saving results

cover_accs = []
avg_counts = []

for rep in range(n_rep):
    netG_A2Bs  = []
    train_graphs_more = []
    empiricals = []
    for lab in present_label:
        ## Create nets for each class
        netG_A2B = models.resnet34(pretrained=False, num_classes=nz)
        netG_A2B = netG_A2B.to(device)
        netG_A2B = nn.DataParallel(netG_A2B)
        netG_B2A = GeneratorB2A(nc, nz, ngf).to(device)
        netG_B2A = nn.DataParallel(netG_B2A)
        netD_A = DiscriminatorA(nc, ndf).to(device)
        netD_A = nn.DataParallel(netD_A)
#         netG_A2B.apply(weights_init)
        netG_B2A.apply(weights_init)
        netD_A.apply(weights_init)

        ## Load the Model directly without training 
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs2, start_epoch, decay_epoch2).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs2, start_epoch, decay_epoch2).step)
        train_graphs = graphs()

        if torch.is_tensor(trainset_A.targets):
            idxs2 = torch.where(trainset_A.targets == lab)[0] 
        else:
            idxs2 = torch.where(torch.Tensor(trainset_A.targets) == lab)[0] 
        train_data2 = torch.utils.data.Subset(trainset_A, idxs2)
        train_loader2  = DataLoader(train_data2, batch_size=batch_size, shuffle=True)
        sample_sizes = [int(len(idxs2) / len(present_label))] * (len(present_label) - 1)
        train_al(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, trainset_A, batch_size, 
                 start_epoch, n_epochs1, nz, lams, device, display, train_graphs,
                 lr_scheduler_G, lr_scheduler_D_A,
                 lab, present_label, all_label, sample_sizes)
        ## get the fake_Bs for each label

        fake_Bs = []
        with torch.no_grad(): 
            for i, batch in enumerate(train_loader2):
                real_A, _ = batch
                fake_B = netG_A2B(real_A.to(device))
                fake_Bs.append(fake_B)
        fake_Bs = torch.cat(fake_Bs)
        ## get the empirical distribution for each label
        empirical = (torch.sum(torch.square(fake_Bs), 1) - nz) / (2 * nz) ** 0.5

        ## get powers to decide new sample sizes
        powers = []
        for cur_lab in present_label:    
            if cur_lab != lab:

                # fake_Cs for this class
                if torch.is_tensor(trainset_A.targets):
                    idxs3 = torch.where(trainset_A.targets == cur_lab)[0] 
                else:
                    idxs3 = torch.where(torch.Tensor(trainset_A.targets) == cur_lab)[0] 
                train_data3 = torch.utils.data.Subset(trainset_A, idxs3)
                train_loader3  = DataLoader(train_data3, batch_size=batch_size, shuffle=False)

                # p_vals_class and fake_Cs store p-values, fake_Cs for each class 
                p_vals_class = torch.zeros(len(idxs3)) 
                fake_Cs = torch.zeros(len(idxs3))

                em_len = len(empirical)

                for i, batch in enumerate(train_loader3):
                    real_A, _ = batch
                    fake_B = netG_A2B(real_A.to(device))
                    fake_C = (torch.sum(torch.square(fake_B), 1) - nz) / (2 * nz) ** 0.5

                    # compute p-value for each sample
                    for j in range(len(fake_C)):
                        p1 = torch.sum(fake_C[j] > empirical) / em_len
                        p2 = torch.sum(fake_C[j] < empirical) / em_len
                        p = 2 * torch.min(p1, p2)
                        # calculate the p-value and put it in the corresponding list
                        p_vals_class[i * batch_size + j] = p.item()
                powers.append(np.sum(np.array(p_vals_class) <= 0.05) / len(idxs3))

        ## train again according to the calculated sample sizes
        sample_sizes = max(powers) - powers + 0.05
        sample_sizes = (sample_sizes / sum(sample_sizes) * len(idxs3)).astype(int)
        train_al(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, trainset_A, batch_size, 
                 n_epochs1, n_epochs2, nz, lams, device, display, train_graphs,
                 lr_scheduler_G, lr_scheduler_D_A,
                 lab, present_label, all_label, sample_sizes)

        fake_Bs = []
        with torch.no_grad(): 
            for i, batch in enumerate(train_loader2):
                real_A, _ = batch
                fake_B = netG_A2B(real_A.to(device))
                fake_Bs.append(fake_B)
        fake_Bs = torch.cat(fake_Bs)
        ## get the empirical distribution for each label
        empirical = (torch.sum(torch.square(fake_Bs), 1) - nz) / (2 * nz) ** 0.5
        empiricals.append(empirical)

        ## save net and graphs for each label
        netG_A2Bs.append(netG_A2B)
        train_graphs_more.append(train_graphs)


    ## get p-values and fake C numbers, visualize them
    p_vals_classes, probs_classes, all_fake_Cs = get_p_and_fake_C(netG_A2Bs, testset_A, batch_size, nz, 
                                                                  present_label, all_label, empiricals, chi)

    cover_acc = torch.zeros(len(all_label))
    avg_count = torch.zeros(len(all_label))
    for i, lab in enumerate(all_label):
        p_vals_class = p_vals_classes[i]
        n = p_vals_class.shape[1]
        cover = 0.0
        counts = 0.0
        if lab in present_label:
            for j in range(n):
                p_set = np.array(present_label)[np.where(p_vals_class[:, j] > 0.05)[0]]
                counts += abs(len(p_set) - 1)
                if lab in p_set:
                    cover += 1
        else:
            for j in range(n):
                pred = np.argmax(p_vals_class[:, j])
                p_set = np.array(present_label)[np.where(p_vals_class[:, j] > 0.05)[0]]
                counts += len(p_set)
                if len(p_set) == 0:
                    cover += 1
        cover_acc[i] = cover / n
        avg_count[i] = counts / n

    print('rep =', rep + 1)
    cover_accs.append(cover_acc)
    avg_counts.append(avg_count)

res = (cover_accs, avg_counts)

import pickle

with open('FCI_Res34_cifar10_OOD.pkl', 'wb') as out:
    pickle.dump(res, out)

tok = datetime.datetime.now()
print('execution time:', tok - tik)
