import datetime
tik = datetime.datetime.now()

import sys
sys.path.insert(1, '../')
import os
os.makedirs('models', exist_ok=True)

import itertools
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader

from models import GeneratorA2B
from models import GeneratorB2A
from models import DiscriminatorA
from models import VGG16

from utils import train_al
from utils import get_p_and_fake_C_fmnist
from utils import weights_init_normal
from utils import weights_init
from utils import LambdaLR

# import different loss functions for GAN B
from geomloss import SamplesLoss

import gc
import numpy as np
import argparse

# hyper-parameters
parser              = argparse.ArgumentParser(description='FCI Fashion-MNIST Training')
parser.add_argument('--net', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'vgg16'], help='network')
parser.add_argument('--n_rep', default=5, type=int, help='number of repetitions')
parser.add_argument('--miss', default=0, type=int, choices=[0, 5, 10], help='missing rate')
parser.add_argument('--nz', default=5, type=int, help='number of latent dimension')
args                = parser.parse_args()

chi                 = True
start_epoch         = 1
# n_epochs1           = 2
# n_epochs2           = 3
n_epochs1           = 125
n_epochs2           = 175
batch_size          = 512
lr_G                = 0.0002
lr_D                = 0.0002
momentum            = 0.9
weight_decay        = 5e-4
## coefficients for GAN X2Z, GAN Z2X, X cycle, and Z cycle
lams                = [1.0, 3.0, 5.0, 5.0, 0.8]
# decay_epoch         = n_epochs//2
decay_epoch2        = n_epochs2//2
lambda_lr           = lambda epoch: 0.5 ** (epoch // 20)
ngf                 = 128
ndf                 = 128
im_size             = 28
nz                  = args.nz
nc                  = 1
ngpu                = torch.cuda.device_count()
cuda                = torch.cuda.is_available()
device              = torch.device("cuda" if cuda else "cpu")
display             = 300

# dataset
transform     = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
trainset_A    = torchvision.datasets.FashionMNIST(root="../datasets",train=True, transform=transform, download=True)
testset_A     = torchvision.datasets.FashionMNIST(root="../datasets",train=False, transform=transform, download=True)

if args.miss > 0:
    missing_label = [9]
    present_label = [x for x in range(10) if x not in missing_label]
else:
    missing_label = []
    present_label = list(range(10))
all_label     = present_label + missing_label
classes       = trainset_A.classes
idxs          = torch.where(torch.Tensor([x in present_label for x in trainset_A.targets]))[0] 
# present_idxs_ = np.where(np.array([x in present_label for x in testset_A.targets]))[0] 
# missing_idxs_ = np.where(np.array([x in missing_label for x in testset_A.targets]))[0] 
train_data    = torch.utils.data.Subset(trainset_A, idxs)

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
avg_errors = []

for rep in range(args.n_rep):
    train_graphs_more = []
    empiricals = []
    for lab in present_label:
        ## Create nets for each class
        if args.net == 'resnet18':
            netG_A2B  = models.resnet18(pretrained=False, num_classes=nz)
            netG_A2B.conv1   = nn.Conv2d(nc, netG_A2B.conv1.weight.shape[0], 3, 1, 1, bias=False) 
            netG_A2B.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
            netG_A2B = netG_A2B.to(device)
            netG_A2B = nn.DataParallel(netG_A2B)
        elif args.net == 'resnet34': 
            netG_A2B  = models.resnet34(pretrained=False, num_classes=nz)
            netG_A2B.conv1   = nn.Conv2d(nc, netG_A2B.conv1.weight.shape[0], 3, 1, 1, bias=False) 
            netG_A2B.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
            netG_A2B = netG_A2B.to(device)
            netG_A2B = nn.DataParallel(netG_A2B)
        elif args.net == 'vgg16':
            netG_A2B  = VGG16(ngpu, nc, nz).to(device)
        
        netG_B2A = GeneratorB2A(ngpu, nc, nz, ngf).to(device)
        netD_A = DiscriminatorA(ngpu, nc, ndf).to(device)
    #     netG_A2B.apply(weights_init)
        netG_B2A.apply(weights_init)
        netD_A.apply(weights_init)

        ## Load the Model directly without training 
        optimizer_G = optim.SGD(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr_G,
                                momentum=momentum,
                                weight_decay=weight_decay)
        optimizer_D_A = optim.SGD(netD_A.parameters(),
                                  lr=lr_D,
                                  momentum=momentum,
                                  weight_decay=weight_decay)

        lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
        lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_lr)
        
#         optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
#                                     lr=lr, betas=(0.5, 0.999))
#         optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
#         lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs2, start_epoch, decay_epoch2).step)
#         lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs2, start_epoch, decay_epoch2).step)
        train_graphs = graphs()

        if torch.is_tensor(trainset_A.targets):
            idxs2 = torch.where(trainset_A.targets == lab)[0] 
        else:
            idxs2 = torch.where(torch.Tensor(trainset_A.targets) == lab)[0] 
        train_data2 = torch.utils.data.Subset(trainset_A, idxs2)
        train_loader2  = DataLoader(train_data2, batch_size=batch_size, shuffle=True)
        sample_sizes = [int(len(idxs2) / len(present_label))] * (len(present_label) - 1)
        train_al(netG_A2B, netG_B2A, netD_A, optimizer_D_A, optimizer_G, trainset_A, batch_size, 
                 start_epoch, n_epochs1, nz, lams, device, 300, train_graphs,
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
                 n_epochs1, n_epochs2, nz, lams, device, 300, train_graphs,
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
        model_save_file = 'models/' + str(args.net) + '_OOD' + str(args.miss) + '_class' + str(lab) + '.pt'
        torch.save(netG_A2B.state_dict(), model_save_file)
        del netG_A2B
        print('Class', lab)

    # In[8]:


    ## get p-values and fake C numbers, visualize them
    p_vals_classes, probs_classes, all_fake_Cs = get_p_and_fake_C_fmnist(args.net, args.miss, testset_A, 512, nz, 
                                                                         present_label, all_label, empiricals, chi)
    
    cover_acc = torch.zeros(len(all_label))
    avg_error = torch.zeros(len(all_label))
    for i, lab in enumerate(all_label):
        p_vals_class = p_vals_classes[i]
        n = p_vals_class.shape[1]
        cover = 0.0
        error = 0.0
        for j in range(n):
            ## sort the p value list and get the corresponding indicies
            sorted = -np.sort(-p_vals_class[:, j])
            indicies = np.argsort(-p_vals_class[:, j])
            if sorted[0] == 0:
                p_set = np.array([])
            else:
                ## find the minimum index when the coverage first exceeds 1-alpha
                idx = np.argmax(np.cumsum(sorted) / np.sum(sorted) > 0.95)
                p_set = indicies[:idx + 1]
            if lab in missing_label:
                error += len(p_set)
                if len(p_set) == 0:
                    cover += 1
            else:
                error += abs(len(p_set) - 1)
                if lab in p_set:
                    cover += 1
        cover_acc[i] = cover / n
        avg_error[i] = error / n
    print('rep =', rep + 1)
    print(cover_acc)
    print(avg_error)
    cover_accs.append(cover_acc)
    avg_errors.append(avg_error)


res = (cover_accs, avg_errors)

import pickle

file_name = 'FCI_' + args.net + '_fmnist_OOD' + str(args.miss) + '.pkl'
with open(file_name, 'wb') as out:
    pickle.dump(res, out)

tok = datetime.datetime.now()
print('execution time:', tok - tik)