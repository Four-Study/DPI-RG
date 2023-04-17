#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools

import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import GeneratorA2B
from models import GeneratorB2A
from models import DiscriminatorA
from models import DiscriminatorB
# from datasets import ConcatDataset

from utils import weights_init_normal
from utils import weights_init
from utils import LambdaLR

import numpy as np
from tqdm import tqdm


# In[2]:


# hyper-parameters
start_epoch         = 0
n_epochs            = 50
batch_size          = 256
lr                  = 0.0002
decay_epoch         = n_epochs//2
ngf                 = 64
ndf                 = 64
im_size             = 28
nz                  = 2
nc                  = 1
ngpu                = 2
cuda                = torch.cuda.is_available()
device              = torch.device("cuda" if cuda else "cpu")

# dataset
# transform = transforms.Compose([transforms.Resize(im_size),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(0.1307,0.3081)])
transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),])
trainset_A    = torchvision.datasets.MNIST(root="./datasets",train=True, transform=transform, download=True)
testset_A     = torchvision.datasets.MNIST(root="./datasets",train=False, transform=transform, download=True)
train_loader  = torch.utils.data.DataLoader(trainset_A, batch_size=batch_size, shuffle=True)
# trainset_B    = torch.randn(len(trainset_A), nz)
# testset_B     = torch.randn(len(testset_A), nz)

# train_loader = torch.utils.data.DataLoader(
#              ConcatDataset(
#                  trainset_A,
#                  trainset_B
#              ),
#              batch_size=batch_size, shuffle=True)


# In[3]:


###### Definition of variables ######
# Networks
netG_A2B = GeneratorA2B(ngpu, nc, nz)
netG_B2A = GeneratorB2A(ngpu, nc, nz, ngf)

netD_A = DiscriminatorA(ngpu, nc, ndf)
netD_B = DiscriminatorB(ngpu, nz)

if cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

# Lossess
# criterion_GAN = torch.nn.MSELoss()
criterion_GAN = torch.nn.BCELoss()
# criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
# optimizer_G = torch.optim.Adam(netG_B2A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

# Loss plot
epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                       12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                       32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                       85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                       225, 245, 268, 293, 320, 350]
cur_epochs          = []

fixed_noise = torch.randn(64, nz, 1, 1, device = device)


# In[4]:


# logger
class graphs:
  def __init__(self):
    self.loss_G_GAN      = []
    self.loss_D          = []
    self.loss_G          = []
    self.loss_G_cycle    = []
train_graphs = graphs()


# In[5]:


###### Training ######
for epoch in range(start_epoch, n_epochs):
    
    pbar = tqdm(total=len(train_loader), position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
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
        
        ###### Discriminator A ######

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)
        loss_D_real.backward()

        # Fake loss
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_fake.backward()

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
#         loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B.squeeze())
        loss_D_real = criterion_GAN(pred_real, target_real)
        loss_D_real.backward()
        
        # Fake loss
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_fake.backward()

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
#         loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B.reshape([bs, nz, 1, 1]))
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A).reshape([bs, nz, 1, 1])
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
        loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
        loss_D = loss_D_A + loss_D_B
        
        pbar.update(1)
        pbar.set_description(
            'Epoch {}: '
            'G Loss: {:.4f}  '
            'D_A Loss: {:.6f}  '
            'D_B Loss: {:.6f}  '.format(epoch + 1,
                                           loss_G.item(),
                                           loss_D_A.item(),
                                           loss_D_B.item()) )

    # progress report
    cur_epochs.append(epoch + 1)
    train_graphs.loss_G.append(loss_G.item())
    train_graphs.loss_G_GAN.append(loss_G_GAN.item())
    train_graphs.loss_G_cycle.append(loss_G_cycle.item())
    train_graphs.loss_D.append(loss_D.item())
    
    if epoch < 5 or (epoch + 1) % 5 == 0:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fake Images")
        fake = netG_B2A(fixed_noise)
        fake = torch.reshape(fake, (64, 1, im_size, im_size))
        plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu()[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step() 


# In[6]:


# display fake numbers
from scipy.sparse.linalg import svds
from scipy.stats import gaussian_kde
nd = min(nz, 4)
densities = [None] * nd
legends = [None] * nd
for i in range(nd):
    nums = fake_B.detach().cpu()[:, i]
    densities[i] = gaussian_kde(nums)
    densities[i].covariance_factor = lambda : .3
    densities[i]._compute_covariance()
    xs = np.linspace(-3,3,200)
    plt.figure(i)
    plt.plot(xs,densities[i](xs))
    legend = ['Dimension {0} out of {1}'.format(i+1, nz)]
    plt.legend(legend)
    plt.show()


# In[7]:


# loss logger
# show the trace plots
# plt.figure(1)
# plt.semilogy(cur_epochs, train_graphs.loss_G)
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Total Loss')

plt.figure(2)
plt.plot(cur_epochs, train_graphs.loss_G)
plt.plot(cur_epochs, train_graphs.loss_G_GAN)
plt.plot(cur_epochs, train_graphs.loss_G_cycle)
plt.plot(cur_epochs, train_graphs.loss_D)
plt.legend(['GAN Total Loss', 'GAN Loss', 'GAN Cycle Loss', 'Discriminator Loss'])
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('cycle GAN Loss')

# plt.figure(3)
# plt.semilogy(cur_epochs, train_graphs.loss_G_cycle)
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Cycle Loss')

# plt.figure(4)
# plt.plot(cur_epochs, train_graphs.loss_D)
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.title('Discriminator Loss')

plt.show()


# ## 
