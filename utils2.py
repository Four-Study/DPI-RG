import random
import time
import datetime
import sys
import matplotlib
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


from geomloss import SamplesLoss

# Lossess
criterion_GAN = torch.nn.BCELoss()
criterion_NLL = torch.nn.GaussianNLLLoss()
criterion_cycle = torch.nn.L1Loss()
criterion_nor = SamplesLoss(loss="gaussian", p = 2, blur = .5)
criterion_was1 = SamplesLoss(loss="sinkhorn", p = 1, blur = .01)
criterion_was2 = SamplesLoss(loss="sinkhorn", p = 2, blur = .01)
criterion_mse = torch.nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()


def train_all(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G_A, optimizer_G_B, 
              train_loader, batch_size, n_epochs, nz, lams, device, display, train_graphs,
              lr_scheduler_G_A, lr_scheduler_G_B, lr_scheduler_D_A, present_label):
    ## fixed noise for display purpose
    fixed_noise   = torch.randn(4*nz, nz, 1, 1, device = device) 
    fixed_noise  += F.one_hot(torch.arange(nz).repeat(4), nz).view(4*nz, nz, 1, 1).to(device) * (nz ** 0.5 + 2)
    N             = len(train_loader.dataset.dataset)
    real_Bs       = torch.randn(N, nz, 1, 1, device = device) 
    
    for epoch in range(1, n_epochs + 1):    
        for batch_idx, (real_A, y, idx) in enumerate(train_loader):
#         for batch_idx, (real_A, y) in enumerate(train_loader):
            # Set model input

            bs = len(real_A)
#             real_B = torch.randn(bs, nz, 1, 1, device = device) 
            real_B = real_Bs[idx, :, :, :]
            real_B += F.one_hot(y, nz).view(bs, nz, 1, 1).to(device) * (nz ** 0.5 + 2)
            real_A = real_A.to(device) 
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
#             fake_B -= F.one_hot(y, nz).view(bs, nz, 1, 1).to(device) * (nz ** 0.5 + 2)
            target_real = torch.ones(bs).to(device)
            target_fake = torch.zeros(bs).to(device)

            ###### Discriminator A ######

            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Generators A2B and B2A ######
            optimizer_G_A.zero_grad()
            optimizer_G_B.zero_grad()

            # GAN loss
#             loss_GAN_A2B = criterion_was2(fake_B, real_B.reshape(bs, nz)) * lams[0]
            loss_GAN_A2B = criterion_mse(fake_B, real_B.view(bs, nz)) * lams[0]
#             var = torch.ones(bs, 1, requires_grad=True, device = device) 
#             loss_GAN_A2B = criterion_NLL(fake_B, real_B.reshape(bs, nz), var) * lams[0]
            

            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lams[1]

            # Cycle loss
            recovered_A = netG_B2A(fake_B.reshape([bs, nz, 1, 1]))
            loss_cycle_ABA = criterion_mse(recovered_A, real_A) * lams[2]

            recovered_B = netG_A2B(fake_A).reshape([bs, nz, 1, 1])
            loss_cycle_BAB = criterion_mse(recovered_B, real_B) * lams[3]

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G_A.step()
            optimizer_G_B.step()
            ###################################

            loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
            loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
            
        # progress report, burning procedure
        if epoch > 2:
            train_graphs.cur_epochs.append(epoch)
            train_graphs.loss_G.append(loss_G.item())
            train_graphs.loss_GAN_A2B.append(loss_GAN_A2B.item())
            train_graphs.loss_GAN_B2A.append(loss_GAN_B2A.item())
            train_graphs.loss_G_GAN.append(loss_G_GAN.item())
            train_graphs.loss_G_cycle.append(loss_G_cycle.item())
            train_graphs.loss_D_A.append(loss_D_A.item())

        if epoch % display == 0:
            print('Epoch {}/{}'.format(epoch, n_epochs), 
                  'GAN A Loss:', loss_GAN_B2A.item(),
                 'GAN B Loss:', loss_GAN_A2B.item(), '\n'
                  'Cycle Loss:', loss_G_cycle.item(),
                 'D_A Loss:', loss_D_A.item())
            plt.figure(figsize=(nz, 5))
            plt.axis("off")
            plt.title("Fake Images")
            fake = netG_B2A(fixed_noise)
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:4*nz], nrow = nz,
                                                     padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
            
        # Update learning rates
        lr_scheduler_G_A.step()
        lr_scheduler_G_B.step()
        lr_scheduler_D_A.step()

def get_p_and_fake_C2(netG_A2B, testset_A, batch_size, nz, 
                      present_label, all_label, empiricals, side):
    device = empiricals[0].device
    em_len = len(empiricals[0])
    const = nz ** (1/2)
    p_vals_classes = []
    all_fake_Cs = []
    
    for lab in all_label:    

        # fake_Cs for this class
        if torch.is_tensor(testset_A.targets):
            idxs_2 = torch.where(testset_A.targets == lab)[0] 
        else:
            idxs_2 = torch.where(torch.Tensor(testset_A.targets) == lab)[0] 
        test_data2 = torch.utils.data.Subset(testset_A, idxs_2)
        test_loader2  = DataLoader(test_data2, batch_size=batch_size, shuffle=False)

        # p_vals_class and fake_Cs store p-values, fake_Cs for each class 
        p_vals_class = torch.zeros(len(present_label), len(idxs_2)) 
        fake_Cs = torch.zeros(len(present_label), len(idxs_2))

        for pidx in range(len(present_label)):
            empirical = empiricals[pidx]
            for i, batch in enumerate(test_loader2):
                real_A, _ = batch
                fake_B = netG_A2B(real_A.to(device))
                fake_B -= F.one_hot(torch.Tensor([present_label[pidx]]).to(device).to(torch.int64), nz) * (nz ** 0.5 + 2)
                fake_C = torch.sum(torch.square(fake_B), 1)
                    
                # compute p-value for each sample
                for j in range(len(fake_C)):
                    if side == 'two-sided':
                        p1 = torch.sum(fake_C[j] > empirical) / em_len
                        p2 = torch.sum(fake_C[j] < empirical) / em_len
                        p = 2 * torch.min(p1, p2)
                    elif side == 'one-sided':
                        p = torch.sum(fake_C[j] < empirical) / em_len
                    # calculate the p-value and put it in the corresponding list
                    p_vals_class[pidx, i * batch_size + j] = p.item()
                    fake_Cs[pidx, i * batch_size + j] = fake_C[j].item()

        p_vals_classes.append(np.array(p_vals_class))
        # concatenate torch data
        all_fake_Cs.append(np.array(fake_Cs))
#         print('Finished Label {}'.format(lab))

    return (p_vals_classes, all_fake_Cs)


def visualize_fake_C(all_fake_Cs, present_label, all_label, missing_label, nz, classes):
    
    print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
    
    # visualization for fake_C which have the test label
    if len(present_label) == 1:
        fig, axs = plt.subplots(1, len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        llim = np.min(all_fake_Cs)
        rlim = np.quantile(all_fake_Cs, 0.9)
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
        llim = np.min(all_fake_Cs)
        rlim = np.quantile(all_fake_Cs, 0.9)
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
    plt.show()
    
    
def visualize_p(p_vals_classes, present_label, all_label, missing_label, nz, classes):

    print('-'*100, '\n', ' ' * 45, 'probabilities', '\n', '-'*100, sep = '')
    # visualization for p-values by class
    if len(present_label) == 1:
        fig, axs = plt.subplots(len(present_label), len(all_label), 
                                figsize=(5*len(all_label), 5*len(present_label)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 

        for i in range(len(all_label)):

            p_vals_class = p_vals_classes[i]

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

            p_vals_class = p_vals_classes[i]

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
#     plt.savefig('size_power.pdf', dpi=150)
    plt.show()
