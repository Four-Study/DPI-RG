import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
import gzip
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.losses import *

# random.seed(1)
# np.random.seed(1)
# matplotlib.use('Agg')

def train_al(netI, netG, netD, optim_I, optim_G, optim_D,
             train_gen, train_loader, batch_size, start_epoch, end_epoch, 
             z_dim, device, lab, present_label, all_label, 
             lambda_gp, lambda_power, lambda_mmd = 3.0, eta = 3, 
             sample_sizes = None, sampled_idxs = None, 
             img_size = 28, nc = 1,
             critic_iter = 15, critic_iter_d = 15, trace=False):

    imbalanced = True
    if sampled_idxs is None:
        imbalanced = False
        sampled_idxs = []
        for cur_lab in present_label:
            if torch.is_tensor(train_gen.targets):
                temp = torch.where(train_gen.targets == cur_lab)[0] 
            else:
                temp = torch.where(torch.Tensor(train_gen.targets) == cur_lab)[0] 
            sampled_idxs.append(temp)
            
    if sample_sizes is None:
        sample_sizes = [int(len(train_loader.dataset.indices) / len(present_label))] * (len(present_label) - 1)

    ## learning rate schedulers
    # scheduler_I = StepLR(optim_I, step_size=15, gamma=0.2)
    # scheduler_G = StepLR(optim_G, step_size=15, gamma=0.2)
    # scheduler_D = StepLR(optim_D, step_size=15, gamma=0.2)
    
    ## training for this label started
    for epoch in range(start_epoch, end_epoch):
        ## first train in null hypothesis
        data = iter(train_loader)
        # 1. Update G, I network
        # (1). Set up parameters of G, I to update
        #      set up parameters of D to freeze
        for p in netD.parameters():
            p.requires_grad = False
        for p in netI.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = True
        # (2). Update G and I
        for _ in range(critic_iter):
            images, _ = next_batch(data, train_loader)
            x = images.view(len(images), nc * img_size ** 2).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            fake_x = netG(z)
            netI.zero_grad()
            netG.zero_grad()
            cost_GI = GI_loss(netI, netG, netD, z, fake_z)
            images, _ = next_batch(data, train_loader)
            x = images.view(len(images), nc * img_size ** 2).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            mmd = mmd_penalty(fake_z, z, kernel="RBF")
            primal_cost = cost_GI + lambda_mmd * mmd
            primal_cost.backward()
            optim_I.step()
            optim_G.step()
        # print('GI: '+str(primal(netI, netG, netD, real_data).cpu().item()))
        if trace:
            print(f'GI: {cost_GI.cpu().item():.6f}')
            print(f'MMD: {lambda_mmd * mmd.cpu().item():.6f}')
        # (3). Append primal and dual loss to list
        # primal_loss_GI.append(primal(netI, netG, netD, z).cpu().item())
        # dual_loss_GI.append(dual(netI, netG, netD, z, fake_z).cpu().item())
        # 2. Update D network
        # (1). Set up parameters of D to update
        #      set up parameters of G, I to freeze
        for p in netD.parameters():
            p.requires_grad = True
        for p in netI.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = False
        # (2). Update D
        for _ in range(critic_iter_d):
            images, _ = next_batch(data, train_loader)
            x = images.view(len(images), nc * img_size ** 2).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            fake_x = netG(z)
            netD.zero_grad()
            cost_D = D_loss(netI, netG, netD, z, fake_z)
            images, y = next_batch(data, train_loader)
            x = images.view(len(images), nc * img_size ** 2).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            gp_D = gradient_penalty_dual(x.data, z.data, netD, netG, netI)
            dual_cost = cost_D + lambda_gp * gp_D
            dual_cost.backward()
            optim_D.step()
            # loss_mmd.append(mmd.cpu().item())
        if trace:
            print(f'D: {cost_D.cpu().item():.6f}')
            print(f'gp: {lambda_gp * gp_D.cpu().item():.6f}')
        # gp.append(gp_D.cpu().item())
        # re.append(primal(netI, netG, netD, z).cpu().item())
        # if (epoch+1) % 5 == 0:
        #     df = pd.DataFrame(fake_z.cpu().numpy())
        #     mvn_result = run_mvn(df)
        #     print(mvn_result[0])
        #     df = pd.DataFrame(z.cpu().numpy())
        #     mvn_result = run_mvn(df)
        #     print(mvn_result[0])
        
        ## then train in alternative hypothesis
        idxs2 = torch.Tensor([])
        count = 0
        for cur_lab in present_label:
            if cur_lab != lab:
                temp = sampled_idxs[cur_lab]
                idxs2 = torch.cat([idxs2, temp[np.random.choice(len(temp), sample_sizes[count], replace=imbalanced)]])
                count += 1
        idxs2 = idxs2.int()
        train_data2 = torch.utils.data.Subset(train_gen, idxs2)
        train_loader2  = DataLoader(train_data2, batch_size=batch_size)

        # 3. Update I network
        # (1). Set up parameters of I to update
        #      set up parameters of G, D to freeze
        for p in netD.parameters():
            p.requires_grad = False
        for p in netI.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False
        
        for i, batch in enumerate(train_loader2):
            x, _ = batch
            bs = len(x)
            
            z = torch.ones(bs, z_dim, 1, 1, device = device) * eta
            x = x.to(device) 
            fake_z = netI(x)

            netI.zero_grad()
            loss_power = lambda_power * I_loss(fake_z, z.reshape(bs, z_dim))
            loss_power.backward()
            optim_I.step()
        if trace:
            print(f'power: {loss_power.cpu().item():.6f}')
        # scheduler_I.step()
        # scheduler_G.step()
        # scheduler_D.step()



def visualize_p(all_p_vals, present_label, all_label, missing_label, nz, classes):

    print('-'*130, '\n', ' ' * 60, 'p-values', '\n', '-'*130, sep = '')
    # visualization for p-values by class
    if len(present_label) == 1:
        fig, axs = plt.subplots(len(present_label), len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 

        for i in range(len(all_label)):

            p_vals_class = all_p_vals[i]

            axs[i].set_xlim([0, 1])
            _ = axs[i].hist(p_vals_class[j, :])
            prop = np.sum(np.array(p_vals_class[j, :] <= 0.05) / len(p_vals_class[j, :]))
            prop = np.round(prop, 4)
            if all_label[i] == present_label[j]:
                axs[i].set_title('Type I Error: {}'.format(prop), fontsize = 20)
            else:
                axs[i].set_title('Power: {}'.format(prop), fontsize = 20)

            if i == 0:
                axs[i].set_ylabel(classes[present_label[j]], fontsize = 25)
            axs[i].set_xlabel(classes[all_label[i]], fontsize = 25)
    else:
        
        fig, axs = plt.subplots(len(present_label), len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 

        for i in range(len(all_label)):

            p_vals_class = all_p_vals[i]

            for j in range(len(present_label)):

                axs[j, i].set_xlim([0, 1])
                _ = axs[j, i].hist(p_vals_class[j, :])
                prop = np.sum(np.array(p_vals_class[j, :] <= 0.05) / len(p_vals_class[j, :]))
                prop = np.round(prop, 4)
                if all_label[i] == present_label[j]:
                    axs[j, i].set_title('Type I Error: {}'.format(prop), fontsize = 20)
                else:
                    axs[j, i].set_title('Power: {}'.format(prop), fontsize = 20)

                if i == 0:
                    axs[j, i].set_ylabel(classes[present_label[j]], fontsize = 25)
                if j == len(present_label) - 1:
                    axs[j, i].set_xlabel(classes[all_label[i]], fontsize = 25)
    fig.supylabel('Training', fontsize = 25)
    fig.supxlabel('Testing', fontsize = 25)
    fig.tight_layout()
    plt.savefig('size_power.pdf', dpi=150)
    plt.show()

def visualize_fake_T(all_fake_Cs, present_label, all_label, missing_label, nz, classes):
    
    print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
    
    # visualization for fake_C which have the test label
    if len(present_label) == 1:
        fig, axs = plt.subplots(1, len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        llim = np.min(all_fake_Cs[0])
        rlim = np.quantile(all_fake_Cs[0], 0.9)
        for i in range(len(all_label)):
            fake_Cs = all_fake_Cs[i]
            axs[i].set_ylabel(classes[all_label[i]], fontsize = 25)
            axs[i].set_xlim([llim, rlim])
            _ = axs[i].hist(fake_Cs[0, :])
            axs[i].set_title('Label {} from Label {}\'s net'.format(all_label[i], present_label[0]), fontsize = 20)

            if i == len(all_label) - 1:
                axs[i].set_xlabel(classes[present_label[0]], fontsize = 25)
    else:
        fig, axs = plt.subplots(len(present_label), len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        llim = np.min(all_fake_Cs[0])
        rlim = np.quantile(all_fake_Cs[0], 0.9)
        for i in range(len(all_label)):
            
            fake_Cs = all_fake_Cs[i]
            
            for j in range(len(present_label)):

                axs[j, i].set_xlim([llim, rlim])
                _ = axs[j, i].hist(fake_Cs[j, :])
                axs[j, i].set_title('Label {} from Label {}\'s net'.format(all_label[i], present_label[j]))

                if i == 0:
                    axs[j, i].set_ylabel(classes[present_label[j]], fontsize = 25)
                if j == len(present_label) - 1:
                    axs[j, i].set_xlabel(classes[all_label[i]], fontsize = 25)
    fig.supylabel('Training', fontsize = 25)
    fig.supxlabel('Testing', fontsize = 25)
    plt.tight_layout()
    plt.savefig('fake_T.pdf', dpi=150)
    plt.show()


def next_batch(data_iter, train_loader):
    try:
        return next(data_iter)
    except StopIteration:
        # Reset the iterator and return the next batch
        return next(iter(train_loader))