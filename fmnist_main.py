## load necessary modules
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.tools import *
from utils.losses import *
from models.mnist import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H%M")

## Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check if gpu is available

## load datasets
# train_gen, dev_gen, test_gen = load(batch_size, batch_size)
# data = inf_train_gen_mnist(train_gen)
transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_gen    = dsets.FashionMNIST(root="./datasets",train=True, transform=transform, download=True)
test_gen     = dsets.FashionMNIST(root="./datasets",train=False, transform=transform, download=True)

## hyper-parameters
n_rep = 1
epochs1 = 50
epochs2 = 50
std = 0.1
learning_rate = 5e-4
weight_decay = 0.01
batch_size = 250
z_dim = 5
lambda_mmd = 3.0
lambda_gp = 0.1
lambda_power = 1.5
eta = 2.5
present_label = list(range(10))
missing_label = []
all_label     = present_label + missing_label
classes       = train_gen.classes

# ************************
# *** DPI-RG Algorithm ***
# ************************

cover_accs = []
avg_errors = []

for rep in range(n_rep):
    T_trains = []
    for lab in present_label:
        ## initialize models
        netI = I_MNIST(nz=z_dim)
        netG = G_MNIST(nz=z_dim)
        netD = D_MNIST(nz=z_dim, power = 6)
        netI = netI.to(device)
        netG = netG.to(device)
        netD = netD.to(device)
        netI = nn.DataParallel(netI)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        # model_save_file = 'mnist_param/' + 'class' + str(lab) + '.pt'
        # netI.load_state_dict(torch.load(model_save_file))
    
        ## set up optimizers
        optim_I = optim.Adam(netI.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optim_G = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optim_D = optim.Adam(netD.parameters(), lr=learning_rate * 5, betas=(0.5, 0.999), 
                             weight_decay=weight_decay)
        ## filter data for each label and train them respectively
        idxs = torch.where(train_gen.targets == lab)[0] 
        train_data = torch.utils.data.Subset(train_gen, idxs)
        train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
        ## train for the first time
        train_al(netI, netG, netD, optim_I, optim_G, optim_D,
                 train_gen, train_loader, batch_size, 0, epochs1, 
                 z_dim, device, lab, present_label, all_label, 
                 lambda_gp, lambda_power, lambda_mmd = lambda_mmd, eta = eta)
    
        ## find out fake_zs
        fake_zs = []
        with torch.no_grad(): 
            for i, batch in enumerate(train_loader):
                x, _ = batch
                fake_z = netI(x.to(device))
                fake_zs.append(fake_z)
        fake_zs = torch.cat(fake_zs)
        ## get the empirical distribution for each label
        T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
    
        ## get powers to determine new sample sizes
        powers = []
        for cur_lab in present_label:    
            if cur_lab != lab:
                # fake_Cs for this class
                if torch.is_tensor(train_gen.targets):
                    idxs3 = torch.where(train_gen.targets == cur_lab)[0] 
                else:
                    idxs3 = torch.where(torch.Tensor(train_gen.targets) == cur_lab)[0] 
                train_data3 = torch.utils.data.Subset(train_gen, idxs3)
                train_loader3  = DataLoader(train_data3, batch_size=batch_size, shuffle=False)
                p_vals = torch.zeros(len(idxs3)) 
                fake_zs = torch.zeros(len(idxs3))
                em_len = len(T_train)
    
                for i, batch in enumerate(train_loader3):
                    x, _ = batch
                    fake_z = netI(x.to(device))
                    T_batch = torch.sqrt(torch.sum(fake_z ** 2, dim=1) + 1)
    
                    # compute p-value for each sample
                    for j in range(len(fake_z)):
                        p1 = torch.sum(T_train > T_batch[j]) / em_len
                        p = p1
                        # calculate the p-value and put it in the corresponding list
                        p_vals[i * batch_size + j] = p.item()
                powers.append(np.sum(np.array(p_vals) <= 0.05) / len(idxs3))
                
        sample_sizes = max(powers) - powers + 0.05
        sample_sizes = (sample_sizes / sum(sample_sizes) * len(idxs3)).astype(int)
        ## train for the second time according to the calculated sample sizes
        train_al(netI, netG, netD, optim_I, optim_G, optim_D,
                 train_gen, train_loader, batch_size, epochs1, epochs2, 
                 z_dim, device, lab, present_label, all_label, 
                 lambda_gp, lambda_power, lambda_mmd = lambda_mmd, 
                 sample_sizes = sample_sizes, eta = eta)
        
        ## find out fake_zs
        fake_zs = []
        with torch.no_grad(): 
            for i, batch in enumerate(train_loader):
                x, _ = batch
                fake_z = netI(x.to(device))
                fake_zs.append(fake_z)
        fake_zs = torch.cat(fake_zs)
        ## get the empirical distribution for each label
        T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
        T_trains.append(T_train)
    
        ## save net and graphs for each label
        model_save_file = f'fmnist_param/{timestamp}_class{lab}.pt'
        torch.save(netI.state_dict(), model_save_file)
        del netI
        # print('Class', lab)
    
    all_p_vals  = []
    all_fake_zs = []
    
    for lab in all_label:    
        if torch.is_tensor(test_gen.targets):
            idxs2 = torch.where(test_gen.targets == lab)[0] 
        else:
            idxs2 = torch.where(torch.Tensor(test_gen.targets) == lab)[0] 
        test_data = torch.utils.data.Subset(test_gen, idxs2)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
        # p_vals and fake_zs store p-values, fake_zs for the current iteration
        fake_zs = torch.zeros(len(present_label), len(idxs2))
        p_vals = torch.zeros(len(present_label), len(idxs2)) 
    
        for pidx in range(len(present_label)):
            T_train = T_trains[pidx]
            em_len = len(T_train)
            netI = I_MNIST(nz=z_dim)
            netI = netI.to(device)
            netI = torch.nn.DataParallel(netI)
            model_save_file = f'fmnist_param/{timestamp}_class{present_label[pidx]}.pt'
            netI.load_state_dict(torch.load(model_save_file))
    
            for i, batch in enumerate(test_loader):
                images, y = batch
                x = images.view(-1, 784).to(device)
                fake_z = netI(x)
                T_batch = torch.sqrt(torch.sum(torch.square(fake_z), 1) + 1) 
                ## compute p-value for each sample
                for j in range(len(fake_z)):
                    p1 = torch.sum(T_train > T_batch[j]) / em_len
                    p = p1
                    # calculate the p-value and put it in the corresponding list
                    p_vals[pidx, i * batch_size + j] = p.item()
    
        all_p_vals.append(np.array(p_vals))
        ## concatenate torch data
        all_fake_zs.append(np.array(fake_zs))
        # print('Finished Label {}'.format(lab))
    
    cover_acc = torch.zeros(len(all_label))
    avg_error = torch.zeros(len(all_label))
    for i, lab in enumerate(all_label):
        p_vals = all_p_vals[i]
        n = p_vals.shape[1]
        cover = 0.0
        error = 0.0
        # counts = 0.0
        for j in range(n):
            pred = np.argmax(p_vals[:, j])
            p_set = np.where(p_vals[:, j] > 0.05)[0]
            # counts += len(p_set)
            if lab in missing_label:
                error += len(p_set)
                if len(p_set) == 0:
                    cover += 1
            else:
                error += abs(len(p_set) - 1)
                if all_label[i] in p_set:
                    cover += 1
        cover_acc[i] = cover / n
        avg_error[i] = error / n
    cover_accs.append(cover_acc)
    avg_errors.append(avg_error)
    print(rep)

res = (cover_accs, avg_errors)

import pickle
filename = f'outputs/FMNIST/resnet18_{timestamp}.pkl'
with open(filename, 'wb') as out:
    pickle.dump(res, out)
