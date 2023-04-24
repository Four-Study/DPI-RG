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
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.models as models
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

# Loss plot

def train(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, train_loader, 
          n_epochs, nz, lam, device, display, train_graphs, lr_scheduler_G, lr_scheduler_D_A):
    ## fixed noise for display purpose
    fixed_noise = torch.randn(64, nz, 1, 1, device = device)
    
    for epoch in range(1, n_epochs + 1):
    
        for i, batch in enumerate(train_loader):
            # Set model input

            real_A, _ = batch
            bs = len(real_A)
#             if bs < train_loader.batch_size:
#                 continue
            real_B = torch.randn(bs, nz, 1, 1, device = device)
            real_A = real_A.to(device) 
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
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
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss
            loss_GAN_A2B = criterion_nor(fake_B, real_B.reshape(bs, nz))
            loss_GAN_A2B *= lam

            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B.reshape([bs, nz, 1, 1]))
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*5.0

            recovered_B = netG_A2B(fake_A).reshape([bs, nz, 1, 1])
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*5.0

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
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
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Fake Images")
            fake = netG_B2A(fixed_noise)
            # fake = torch.reshape(fake, (64, nc, real_A.size()[2], real_A.size()[2]))
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()

def train_al(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, trainset_A, batch_size, 
             n_epochs1, n_epochs2, nz, lams, device, display, train_graphs, 
             lr_scheduler_G, lr_scheduler_D_A,
             lab, present_label, all_label, sample_sizes):
    ## fixed noise for display purpose
    fixed_noise = torch.randn(32, nz, 1, 1, device = device)
    
    ## first train in null hypothesis
    ## filter data for each label and train them respectively
    if torch.is_tensor(trainset_A.targets):
        idxs = torch.where(trainset_A.targets == lab)[0] 
    else:
        idxs = torch.where(torch.Tensor(trainset_A.targets) == lab)[0] 
    train_data = torch.utils.data.Subset(trainset_A, idxs)
    train_loader  = DataLoader(train_data, batch_size=batch_size)
    for epoch in range(n_epochs1, n_epochs2):

        for i, batch in enumerate(train_loader):
            # Set model input

            real_A, _ = batch
            bs = len(real_A)
            real_B = torch.randn(bs, nz, 1, 1, device = device)
            real_A = real_A.to(device) 
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
            target_real = torch.ones(bs).to(device)
            target_fake = torch.zeros(bs).to(device)

            ######### Discriminator A #########

            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss 
#             loss_GAN_A2B = criterion_was2(fake_B, real_B.reshape(bs, nz)) * lams[0]
            var = torch.ones(bs, 1, requires_grad=True, device = device) 
            loss_GAN_A2B = criterion_NLL(fake_B, real_B.reshape(bs, nz), var) * lams[0]
#             loss_GAN_A2B = criterion_nor(fake_B, real_B.reshape(bs, nz)) * lams[0]

            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lams[1]

            # Cycle loss
            recovered_A = netG_B2A(fake_B.reshape([bs, nz, 1, 1]))
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * lams[2]

            recovered_B = netG_A2B(fake_A).reshape([bs, nz, 1, 1])
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * lams[3]

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
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
            print('Epoch {}/{}'.format(epoch, n_epochs2), 
                  'GAN A Loss:', round(loss_GAN_B2A.item(), 6),
                 'GAN B Loss:', round(loss_GAN_A2B.item(), 6), '\n'
                  'Cycle Loss:', round(loss_G_cycle.item(), 6),
                 'D_A Loss:', round(loss_D_A.item(), 6))
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Fake Images")
            fake = netG_B2A(fixed_noise)
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:32], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
        
        ## then train in alternative hypothesis
        idxs2 = torch.Tensor([])
        count = 0
        for cur_lab in present_label:
            if cur_lab != lab:
                if torch.is_tensor(trainset_A.targets):
                    temp = torch.where(trainset_A.targets == cur_lab)[0] 
                else:
                    temp = torch.where(torch.Tensor(trainset_A.targets) == cur_lab)[0] 
                idxs2 = torch.cat([idxs2, temp[np.random.choice(len(temp), sample_sizes[count], replace=False)]])
                count += 1
        idxs2 = idxs2.int()
        train_data2 = torch.utils.data.Subset(trainset_A, idxs2)
        train_loader2  = DataLoader(train_data2, batch_size=batch_size)

#         if (epoch % 100 == 0):
#             eta /= 10
        for i, batch in enumerate(train_loader2):
            # Set model input

            real_A, _ = batch
            bs = len(real_A)
            
            real_B = torch.ones(bs, nz, 1, 1, device = device) * 3
            real_A = real_A.to(device) 
            fake_B = netG_A2B(real_A)

            ###### Generators A2B ######
            optimizer_G.zero_grad()

            # GAN loss
#             loss_GAN_A2B = (criterion_l1(fake_B, real_B.reshape(bs, nz)) - 10*nz) * lams[4] 
            loss_GAN_A2B = criterion_mse(fake_B, real_B.reshape(bs, nz)) * lams[4] 
            loss_GAN_A2B.backward()

            optimizer_G.step()
            ###################################

            loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
            loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
            
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        
def get_p_and_fake_C_fmnist(net, miss, testset_A, batch_size, nz, 
                            present_label, all_label, empiricals, chi=True):
    device = empiricals[0].device
    const = nz ** (1/2)
    p_vals_classes = []
    probs_classes = []
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
        probs_class = torch.zeros(len(present_label), len(idxs_2)) 
        fake_Cs = torch.zeros(len(present_label), len(idxs_2))

        for pidx in range(len(present_label)):
            em_len = len(empiricals[pidx])
            if net == 'resnet18':
                netG_A2B  = models.resnet18(pretrained=False, num_classes=nz)
                netG_A2B.conv1   = nn.Conv2d(1, netG_A2B.conv1.weight.shape[0], 3, 1, 1, bias=False) 
                netG_A2B.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                netG_A2B = netG_A2B.to(device)
                netG_A2B = nn.DataParallel(netG_A2B)
            elif net == 'resnet34': 
                netG_A2B  = models.resnet34(pretrained=False, num_classes=nz)
                netG_A2B.conv1   = nn.Conv2d(1, netG_A2B.conv1.weight.shape[0], 3, 1, 1, bias=False) 
                netG_A2B.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
                netG_A2B = netG_A2B.to(device)
                netG_A2B = nn.DataParallel(netG_A2B)
            elif net == 'vgg16':
                netG_A2B  = VGG16(ngpu, 1, nz).to(device)
            model_save_file = 'models/' + str(net) + '_OOD' + str(miss) + '_class' + str(present_label[pidx]) + '.pt'
            netG_A2B.load_state_dict(torch.load(model_save_file))

            for i, batch in enumerate(test_loader2):
                real_A, _ = batch
                fake_B = netG_A2B(real_A.to(device))
                if chi:
                    fake_C = (torch.sum(torch.square(fake_B), 1) - nz) / (2 * nz) ** 0.5
                else:
                    fake_C = torch.mul(torch.mean(fake_B, 1), const)
                    
                # compute p-value for each sample
                for j in range(len(fake_C)):
                    p1 = torch.sum(empiricals[pidx] > fake_C[j]) / em_len
#                     p2 = torch.sum(fake_C[j] < empiricals[pidx]) / em_len
#                     p = 2 * torch.min(p1, p2)
                    p = p1
                    # calculate the p-value and put it in the corresponding list
                    p_vals_class[pidx, i * batch_size + j] = p.item()
                    fake_Cs[pidx, i * batch_size + j] = fake_C[j].item()
        
        for i in range(len(idxs_2)):
            probs_class[:, i] = p_vals_class[:, i] / torch.sum(p_vals_class[:, i])

        p_vals_classes.append(np.array(p_vals_class))
        probs_classes.append(np.array(probs_class))
        # concatenate torch data
        all_fake_Cs.append(np.array(fake_Cs))
#         print('Finished Label {}'.format(lab))
    
    return (p_vals_classes, probs_classes, all_fake_Cs)


def get_p_and_fake_C(net, miss, testset_A, batch_size, nz, 
                     present_label, all_label, empiricals, chi=True):
    device = empiricals[0].device
    const = nz ** (1/2)
    p_vals_classes = []
    probs_classes = []
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
        probs_class = torch.zeros(len(present_label), len(idxs_2)) 
        fake_Cs = torch.zeros(len(present_label), len(idxs_2))

        for pidx in range(len(present_label)):
            em_len = len(empiricals[pidx])
            if net == 'resnet18':
                netG_A2B  = models.resnet18(pretrained=False, num_classes=nz)
            elif net == 'resnet34': 
                netG_A2B  = models.resnet34(pretrained=False, num_classes=nz)
            elif net == 'vgg16':
                netG_A2B  = models.vgg16(pretrained=False, num_classes=nz)
#             netG_A2B = netG_A2Bs[pidx]
            netG_A2B = netG_A2B.to(device)
            netG_A2B = torch.nn.DataParallel(netG_A2B)
            model_save_file = 'models/' + str(net) + '_OOD' + str(miss) + '_class' + str(present_label[pidx]) + '.pt'
            netG_A2B.load_state_dict(torch.load(model_save_file))

            for i, batch in enumerate(test_loader2):
                real_A, _ = batch
                fake_B = netG_A2B(real_A.to(device))
                if chi:
                    fake_C = (torch.sum(torch.square(fake_B), 1) - nz) / (2 * nz) ** 0.5
                else:
                    fake_C = torch.mul(torch.mean(fake_B, 1), const)
                    
                # compute p-value for each sample
                for j in range(len(fake_C)):
                    p1 = torch.sum(empiricals[pidx] > fake_C[j]) / em_len
#                     p2 = torch.sum(fake_C[j] < empiricals[pidx]) / em_len
#                     p = 2 * torch.min(p1, p2)
                    p = p1
                    # calculate the p-value and put it in the corresponding list
                    p_vals_class[pidx, i * batch_size + j] = p.item()
                    fake_Cs[pidx, i * batch_size + j] = fake_C[j].item()
        
        for i in range(len(idxs_2)):
            probs_class[:, i] = p_vals_class[:, i] / torch.sum(p_vals_class[:, i])

        p_vals_classes.append(np.array(p_vals_class))
        probs_classes.append(np.array(probs_class))
        # concatenate torch data
        all_fake_Cs.append(np.array(fake_Cs))
#         print('Finished Label {}'.format(lab))
    
    return (p_vals_classes, probs_classes, all_fake_Cs)


def train_all(netG_A2B, netG_B2A, netD_A, netD_B, optimizer_D_A, optimizer_D_B, optimizer_G, 
              train_loader, batch_size, n_epochs, nz, lams, device, display, train_graphs,
              lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B, present_label):
    ## fixed noise for display purpose
    nclass        = len(present_label)
    fixed_noise   = torch.randn(4*nclass, nz*nclass, device = device) 
    add           = F.one_hot(torch.arange(nclass).repeat(4), nclass).repeat_interleave(nz, dim = 1)
    fixed_noise  += add.to(device) * (nclass ** 0.5 + 2)
    
    for epoch in range(1, n_epochs + 1):    

        losses_GAN_B2A = []
        losses_GAN_A2B = []
        losses_G_cycle = []
        losses_D_A = []
        losses_D_B = []

        for batch_idx, (real_A, y) in enumerate(train_loader):
            # Set model input

            bs = len(real_A)
            real_N = torch.randn(bs, nz*nclass, 1, 1, device = device) 
            add    = F.one_hot(y, nclass).repeat_interleave(nz, dim = 1).view(bs, nclass*nz, 1, 1)
            real_B = real_N + add.to(device) * (nclass ** 0.5 + 2)
            real_A = real_A.to(device) 
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
            minus  = F.one_hot(y, nclass).repeat_interleave(nz, dim = 1).view(bs, nclass*nz)
            fake_N = fake_B - minus.to(device) * (nclass ** 0.5 + 2)
            target_real = torch.ones(bs).to(device)
            target_fake = torch.zeros(bs).to(device)

            ######### Discriminator A #########

            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_A_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_A_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_A_real + loss_D_A_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################
            
            ######### Discriminator B #########

            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_N.reshape(bs, nz*nclass))
            loss_D_B_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_B(fake_N.detach())
            loss_D_B_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_B = (loss_D_B_real + loss_D_B_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss
    
            pred_fake = netD_B(fake_N)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * lams[0]

            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lams[1]
            
            # Cycle loss

            recovered_A = netG_B2A(fake_B.reshape([bs, nz*nclass, 1, 1]))
            loss_cycle_ABA = criterion_mse(recovered_A, real_A) * lams[2]

            recovered_B = netG_A2B(fake_A).reshape([bs, nz*nclass, 1, 1])
            loss_cycle_BAB = criterion_mse(recovered_B, real_B) * lams[3]
            
            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
            loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
            
            
            losses_GAN_B2A.append(loss_GAN_B2A.item())
            losses_GAN_A2B.append(loss_GAN_A2B.item())
            losses_G_cycle.append(loss_G_cycle.item())
            losses_D_A.append(loss_D_A.item())
            losses_D_B.append(loss_D_B.item())
            
        losses_GAN_B2A = np.array(losses_GAN_B2A)
        losses_GAN_A2B = np.array(losses_GAN_A2B)
        losses_G_cycle = np.array(losses_G_cycle)
        losses_D_A = np.array(losses_D_A)
        losses_D_B = np.array(losses_D_B)
            
        # progress report, burning procedure
        if epoch > 2:
            train_graphs.cur_epochs.append(epoch)
            train_graphs.loss_G.append(loss_G.item())
            train_graphs.loss_GAN_A2B.append(loss_GAN_A2B.item())
            train_graphs.loss_GAN_B2A.append(loss_GAN_B2A.item())
            train_graphs.loss_G_GAN.append(loss_G_GAN.item())
            train_graphs.loss_G_cycle.append(loss_G_cycle.item())
            train_graphs.loss_D_A.append(loss_D_A.item())
            train_graphs.loss_D_B.append(loss_D_B.item())

        if epoch % display == 0:
            print('Epoch {}/{}'.format(epoch, n_epochs), 
                  'GAN A Loss:', losses_GAN_B2A.mean(),
                 'GAN B Loss:', losses_GAN_A2B.mean(), '\n'
                  'Cycle Loss:', losses_G_cycle.mean(),
                 'D_A Loss:', losses_D_A.mean(),
                 'D_B Loss:', losses_D_B.mean())
            plt.figure(figsize=(nclass, 4))
            plt.axis("off")
            plt.title("Fake Images")
            fake = netG_B2A(fixed_noise.view([4*nclass, nz*nclass, 1, 1]))
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:4*nclass], nrow = nclass,
                                                     padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
            
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

# def train_all(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, 
#               train_loader, batch_size, n_epochs, nz, lams, device, display, train_graphs,
#               lr_scheduler_G, lr_scheduler_D_A, present_label):
#     ## fixed noise for display purpose
#     nclass        = len(present_label)
#     fixed_noise   = torch.randn(4*nclass, nz*nclass, device = device) 
#     add_on        = F.one_hot(torch.arange(nclass).repeat(4), nclass).repeat_interleave(nz, dim = 1)
#     fixed_noise  += add_on.to(device) * (nclass ** 0.5 + 2)
    
#     for epoch in range(1, n_epochs + 1):    

#         losses_GAN_B2A = []
#         losses_GAN_A2B = []
#         losses_G_cycle = []
#         losses_D_A = []

#         for batch_idx, (real_A, y) in enumerate(train_loader):
#             # Set model input

#             bs = len(real_A)
#             real_B = torch.randn(bs, nz*nclass, 1, 1, device = device) 
#             add_on = F.one_hot(y, nclass).repeat_interleave(nz, dim = 1).view(bs, nclass*nz, 1, 1)
#             real_B += add_on.to(device) * (nclass ** 0.5 + 2)
#             real_A = real_A.to(device) 
#             fake_A = netG_B2A(real_B)
#             fake_B = netG_A2B(real_A)
#             target_real = torch.ones(bs).to(device)
#             target_fake = torch.zeros(bs).to(device)

#             ###### Discriminator A ######

#             optimizer_D_A.zero_grad()

#             # Real loss
#             pred_real = netD_A(real_A)
#             loss_D_real = criterion_GAN(pred_real, target_real)
#             # Fake loss
#             pred_fake = netD_A(fake_A.detach())
#             loss_D_fake = criterion_GAN(pred_fake, target_fake)
#             # Total loss
#             loss_D_A = (loss_D_real + loss_D_fake)*0.5
#             loss_D_A.backward()

#             optimizer_D_A.step()
#             ###################################

#             ###### Generators A2B and B2A ######
#             optimizer_G.zero_grad()

#             # GAN loss
# #             loss_GAN_A2B = criterion_was2(fake_B, real_B.reshape(bs, nz)) * lams[0]
# #             loss_GAN_A2B = criterion_nor(fake_B, real_B.view(bs, nz)) * lams[0]
# #             var = torch.ones(bs, 1, requires_grad=True, device = device) 
# #             loss_GAN_A2B_1 = criterion_NLL(fake_B, real_B.reshape(bs, nz*nclass), var)
# #             loss_GAN_A2B_2 = criterion_mse(fake_B[:, y], real_B.reshape(bs, nz*nclass)[:, y])
#             loss_GAN_A2B_2 = criterion_mse(fake_B, real_B.reshape(bs, nz*nclass))
# #             loss_GAN_A2B = (0.8 * loss_GAN_A2B_1 + 0.2 * loss_GAN_A2B_2) * lams[0]
#             loss_GAN_A2B = loss_GAN_A2B_2 * lams[0]


#             pred_fake = netD_A(fake_A)
#             loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lams[1]
            
#             # Cycle loss

#             recovered_A = netG_B2A(fake_B.reshape([bs, nz*nclass, 1, 1]))
#             loss_cycle_ABA = criterion_mse(recovered_A, real_A) * lams[2]

#             recovered_B = netG_A2B(fake_A).reshape([bs, nz*nclass, 1, 1])
#             loss_cycle_BAB = criterion_mse(recovered_B, real_B) * lams[3]
            
            

#             # Total loss
#             loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
#             loss_G.backward()

#             optimizer_G.step()
#             ###################################

#             loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
#             loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
            
            
#             losses_GAN_B2A.append(loss_GAN_B2A.item())
#             losses_GAN_A2B.append(loss_GAN_A2B.item())
#             losses_G_cycle.append(loss_G_cycle.item())
#             losses_D_A.append(loss_D_A.item())
            
#         losses_GAN_B2A = np.array(losses_GAN_B2A)
#         losses_GAN_A2B = np.array(losses_GAN_A2B)
#         losses_G_cycle = np.array(losses_G_cycle)
#         losses_D_A = np.array(losses_D_A)
            
#         # progress report, burning procedure
#         if epoch > 2:
#             train_graphs.cur_epochs.append(epoch)
#             train_graphs.loss_G.append(loss_G.item())
#             train_graphs.loss_GAN_A2B.append(loss_GAN_A2B.item())
#             train_graphs.loss_GAN_B2A.append(loss_GAN_B2A.item())
#             train_graphs.loss_G_GAN.append(loss_G_GAN.item())
#             train_graphs.loss_G_cycle.append(loss_G_cycle.item())
#             train_graphs.loss_D_A.append(loss_D_A.item())

#         if epoch % display == 0:
#             print('Epoch {}/{}'.format(epoch, n_epochs), 
#                   'GAN A Loss:', losses_GAN_B2A.mean(),
#                  'GAN B Loss:', losses_GAN_A2B.mean(), '\n'
#                   'Cycle Loss:', losses_G_cycle.mean(),
#                  'D_A Loss:', losses_D_A.mean())
#             plt.figure(figsize=(nclass, 4))
#             plt.axis("off")
#             plt.title("Fake Images")
#             fake = netG_B2A(fixed_noise.view([4*nclass, nz*nclass, 1, 1]))
#             plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:4*nclass], nrow = nclass,
#                                                      padding=2, normalize=True).cpu(),(1,2,0)))
#             plt.show()
            
#         # Update learning rates
#         lr_scheduler_G.step()
#         lr_scheduler_D_A.step()

def get_p_and_fake_C2(netG_A2B, testset_A, batch_size, nz, 
                      present_label, all_label, empiricals, side):
    nclass = len(present_label)
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

#         print(len(idxs_2))
        # p_vals_class and fake_Cs store p-values, fake_Cs for each class 
        p_vals_class = torch.zeros(len(present_label), len(idxs_2)) 
        fake_Cs = torch.zeros(len(present_label), len(idxs_2))

        for pidx in range(len(present_label)):
            empirical = empiricals[pidx]
            for i, batch in enumerate(test_loader2):
                real_A, _ = batch
                bs     = len(real_A)
                fake_B = netG_A2B(real_A.to(device))
                fake_B -= F.one_hot(torch.Tensor([present_label[pidx]] * bs).to(device).to(torch.int64), nclass).repeat_interleave(nz, dim = 1) * (nclass ** 0.5 + 2)
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
    
def conformal_p_vals(all_label, missing_label, p_vals_classes, method):
    
    print("Class", "Pred Accuracy", "Coverage Prob", "Average Count")
    
    if method == "individual_p":
        for i, lab in enumerate(all_label):
            p_vals_class = p_vals_classes[i]
            n = p_vals_class.shape[1]
            correct = 0.0
            cover = 0.0
            counts = 0.0
            for j in range(n):
                pred = np.argmax(p_vals_class[:, j])
                p_set = np.where(p_vals_class[:, j] > 0.05)[0]
                counts += len(p_set)
                if lab in missing_label:
                    if len(p_set) == 0:
                        correct += 1
                        cover += 1
                else:
                    if all_label[i] == pred:
                        correct += 1
                    if all_label[i] in p_set:
                        cover += 1
            pred_acc = correct / n
            cover_acc = cover / n
            avg_count = counts / n
            print(i, " "*4, pred_acc, " "*6, cover_acc, " "*10, avg_count)
    elif method == "bonferroni":
        for i in range(len(all_label)):
            p_vals_class = p_vals_classes[i]
            n = p_vals_class.shape[1]
            correct = 0.0
            cover = 0.0
            counts = 0.0
            for j in range(n):
                pred = np.argmax(p_vals_class[:, j])
                # sorted index and cumulative summation
                sort_idx = np.argsort(p_vals_class[:, j])
                rej_num = sum(np.cumsum(p_vals_class[sort_idx, j]) <= 0.05*10)

                p_set = sort_idx[rej_num:]
                counts += len(p_set)
                if lab in missing_label:
                    if len(p_set) == 0:
                        correct += 1
                        cover += 1
                else:
                    if all_label[i] == pred:
                        correct += 1
                    if all_label[i] in p_set:
                        cover += 1
            pred_acc = correct / n
            cover_acc = cover / n
            avg_count = counts / n
            print(i, " "*4, pred_acc, " "*6, cover_acc, " "*10, avg_count)
  
    
# def conformal_probs(all_label, probs_classes):
#     print("Class", "Pred Accuracy", "Coverage Prob", "Average Count")
#     for i in range(len(all_label)):
#         probs_class = probs_classes[i]
#         n = probs_class.shape[1]
#         correct = 0.0
#         cover = 0.0
#         counts = 0.0
#         for j in range(n):
#             pred = np.argmax(probs_class[:, j])
#             # sorted index and cumulative summation
#             sort_idx = np.argsort(-probs_class[:, j])
#             acp_num = sum(np.cumsum(probs_class[sort_idx, j]) <= 0.95)

#             p_set = sort_idx[:acp_num]
#             counts += len(p_set)
#             if i == pred:
#                 correct += 1
#             if i in p_set:
#                 cover += 1
#         pred_acc = correct / n
#         cover_acc = cover / n
#         avg_count = counts / n
#         print(i, " "*4, pred_acc, " "*6, cover_acc, " "*10, avg_count)