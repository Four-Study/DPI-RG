import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
# import urllib
# import gzip
# import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .losses import *
from .mnist import I_MNIST, G_MNIST, D_MNIST


class DPI:
    def __init__(self, z_dim, lr_GI, lr_D, weight_decay, batch_size, epochs1, epochs2, lambda_mmd, lambda_gp, lambda_power, eta,
                 present_label, missing_label = [], img_size=28, nc=1, critic_iter=10, critic_iter_d=10, critic_iter_p=10, decay_epochs=None, gamma=0.2, device=None, timestamp=None):
        self.z_dim = z_dim
        self.lr_GI = lr_GI
        self.lr_D = lr_D
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.lambda_mmd = lambda_mmd
        self.lambda_gp = lambda_gp
        self.lambda_power = lambda_power
        self.eta = eta
        self.img_size = img_size
        self.nc = nc
        self.critic_iter = critic_iter
        self.critic_iter_d = critic_iter_d
        self.critic_iter_p = critic_iter_p
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_gen = dsets.FashionMNIST(root="./datasets", train=True, transform=self.transform, download=True)
        self.test_gen = dsets.FashionMNIST(root="./datasets", train=False, transform=self.transform, download=True)
        self.present_label = present_label
        self.missing_label = missing_label
        self.all_label = self.present_label + self.missing_label
        self.T_trains = []

        # Set timestamp and mode
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            self.inference_only = False  # Indicates we are in training mode
        else:
            self.timestamp = timestamp
            self.inference_only = True  # Indicates we are in inference mode

        # Initialize or load models based on the mode
        self.setup_models()

    def setup_models(self):
        """Setup models based on the mode (training or inference)."""
        if self.inference_only:
            print("Setting up models for inference mode.")
            for label in self.present_label:
                self.load_inference_model(label)  # Load "I" model for inference
        else:
            print("Setting up models for training mode.")
            for label in self.present_label:
                self.initialize_model(label)  # Initialize "I", "G", "D" for training

    def initialize_model(self, label):
        """Initialize 'I', 'G', 'D' models for a given label for training."""
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netG = G_MNIST(nz=self.z_dim).to(self.device)
        netD = D_MNIST(nz=self.z_dim).to(self.device)
        netI, netG, netD = nn.DataParallel(netI), nn.DataParallel(netG), nn.DataParallel(netD)

        self.models[label] = {'I': netI, 'G': netG, 'D': netD}
        self.optimizers[label] = {
            'I': optim.Adam(netI.parameters(), lr=self.lr_GI, betas=(0.5, 0.999)),
            'G': optim.Adam(netG.parameters(), lr=self.lr_GI, betas=(0.5, 0.999)),
            'D': optim.Adam(netD.parameters(), lr=self.lr_D, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        }

    def load_inference_model(self, label):
        """Load the pre-trained 'I' model for inference."""
        model_save_file = f'fmnist_param/{self.timestamp}_class{label}.pt'
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netI = nn.DataParallel(netI)
        try:
            netI.load_state_dict(torch.load(model_save_file))
            self.models[label] = {'I': netI}
            print(f"Successfully loaded model for label {label} from {model_save_file}.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model for label {label} at {model_save_file}. Error: {e}")


    def train(self):
        """Train models for each present label."""
        if self.inference_only:
            print("Training is not allowed when a timestamp is provided. Exiting the train method.")
            return

        for label in self.present_label:
            print(f"{'-'*100}\nStart to train label: {label}\n{'-'*100}")

            train_loader = self.get_data_loader(label)

            # Retrieve models and optimizers for the current label
            netI, netG, netD = self.models[label]['I'], self.models[label]['G'], self.models[label]['D']
            optim_I, optim_G, optim_D = self.optimizers[label]['I'], self.optimizers[label]['G'], self.optimizers[label]['D']
            
            # Initialize lists to store losses
            GI_losses, MMD_losses, D_losses, GP_losses, Power_losses = [], [], [], [], []

            # First round of training
            self.train_label(
                label, netI, netG, netD, optim_I, optim_G, optim_D,
                train_loader, 0, self.epochs1,
                sample_sizes=None, trace=True,
                GI_losses=GI_losses, MMD_losses=MMD_losses,
                D_losses=D_losses, GP_losses=GP_losses, Power_losses=Power_losses
            )

            # Get fake_zs for further training or analysis
            fake_zs = self.get_fake_zs(label, train_loader)

            # Calculate empirical distribution T_train
            T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)

            # Compute powers and new sample sizes
            sample_sizes = self.compute_powers_and_sizes(T_train, label)

            # Freeze batch normalization layers before the second round of training
            self.freeze_batch_norm_layers(netI)

            # Second round of training with new sample sizes
            self.train_label(
                label, netI, netG, netD, optim_I, optim_G, optim_D,
                train_loader, self.epochs1, self.epochs2,
                sample_sizes=sample_sizes, trace=True,
                GI_losses=GI_losses, MMD_losses=MMD_losses,
                D_losses=D_losses, GP_losses=GP_losses, Power_losses=Power_losses
            )
            # Save the trained model
            self.save_model(label)

            # Get fake_zs for further training or analysis
            fake_zs = self.get_fake_zs(label, train_loader)

            # Calculate empirical distribution T_train
            T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
            self.T_trains.append(T_train)

            # Save the loss plots to the graphs folder
            self.save_loss_plots(label, GI_losses, MMD_losses, D_losses, GP_losses, Power_losses)
            print(f"{'-'*100}\nFinish to train label: {label}\n{'-'*100}")

    def train_label(self, label, netI, netG, netD, optim_I, optim_G, optim_D, train_loader, start_epoch, end_epoch, 
            sample_sizes=None, sampled_idxs=None, trace=False):
        
        # ## set the models to train mode
        # netI.train()
        # netG.train()
        # netD.train()

        imbalanced = True
        if sampled_idxs is None:
            imbalanced = False
            sampled_idxs = []
            for cur_lab in self.present_label:
                if torch.is_tensor(self.train_gen.targets):
                    temp = torch.where(self.train_gen.targets == cur_lab)[0] 
                else:
                    temp = torch.where(torch.Tensor(self.train_gen.targets) == cur_lab)[0] 
                sampled_idxs.append(temp)
                
        if sample_sizes is None:
            sample_sizes = [int(len(train_loader.dataset.indices) / len(self.present_label))] * (len(self.present_label) - 1)
            
        # Learning rate schedulers
        if isinstance(self.decay_epochs, int):
            scheduler_I = StepLR(optim_I, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_G = StepLR(optim_G, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_D = StepLR(optim_D, step_size=self.decay_epochs, gamma=self.gamma)
        
        # Training for this label started
        for epoch in range(start_epoch, end_epoch):
            print(f'Epoch = {epoch}')
            data = iter(train_loader)
            
            # 1. Update G, I network
            for p in netD.parameters():
                p.requires_grad = False
            for p in netI.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = True
                
            for _ in range(self.critic_iter):
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                fake_z = netI(x)
                # fake_x = netG(z)
                netI.zero_grad()
                netG.zero_grad()
                cost_GI = GI_loss(netI, netG, netD, z, fake_z)
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                fake_z = netI(x)
                mmd = mmd_penalty(fake_z, z, kernel="RBF")
                primal_cost = cost_GI + self.lambda_mmd * mmd
                primal_cost.backward()
                optim_I.step()
                optim_G.step()
            if trace:
                print(f'GI loss:  {cost_GI.cpu().item():.6f}')
                print(f'MMD loss: {self.lambda_mmd * mmd.cpu().item():.6f}')
            
            # 2. Update D network
            for p in netD.parameters():
                p.requires_grad = True
            for p in netI.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False
                
            for _ in range(self.critic_iter_d):
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                fake_z = netI(x)
                netD.zero_grad()
                cost_D = D_loss(netI, netG, netD, z, fake_z)
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                gp_D = gradient_penalty_dual(x.data, z.data, netD, netG, netI)
                dual_cost = cost_D + self.lambda_gp * gp_D
                dual_cost.backward()
                optim_D.step()
            if trace:
                print(f'D loss:   {cost_D.cpu().item():.6f}')
                print(f'gp loss:  {self.lambda_gp * gp_D.cpu().item():.6f}')
            
            # Train in alternative hypothesis
            idxs2 = torch.Tensor([])
            count = 0
            for cur_lab in self.present_label:
                if cur_lab != label:
                    temp = sampled_idxs[cur_lab]
                    idxs2 = torch.cat([idxs2, temp[np.random.choice(len(temp), sample_sizes[count], replace=imbalanced)]])
                    count += 1
            idxs2 = idxs2.int()
            train_data2 = torch.utils.data.Subset(self.train_gen, idxs2)
            train_loader2 = DataLoader(train_data2, batch_size=self.batch_size)
            data2 = iter(train_loader2)
            
            # 3. Update I network
            for p in netD.parameters():
                p.requires_grad = False
            for p in netI.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False
                
            for _ in range(self.critic_iter_p):
                images, _ = self.next_batch(data2, train_loader2)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                bs = len(x)
                
                z = torch.ones(bs, self.z_dim, 1, 1, device=self.device) * self.eta
                x = x.to(self.device)
                fake_z = netI(x)

                netI.zero_grad()
                loss_power = self.lambda_power * I_loss(fake_z, z.reshape(bs, self.z_dim))
                loss_power.backward()
                optim_I.step()
            if isinstance(self.decay_epochs, int):
                scheduler_I.step()
                scheduler_G.step()
                scheduler_D.step()
            if trace:
                print(f'power loss: {loss_power.cpu().item():.6f}')

    def get_data_loader(self, label):
        idxs = torch.where(self.train_gen.targets == label)[0]
        train_data = torch.utils.data.Subset(self.train_gen, idxs)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

    def get_fake_zs(self, label, train_loader):
        netI = self.models[label]['I']
        netI.eval()  # Set the model to evaluation mode
        fake_zs = []
        with torch.no_grad():
            for x, _ in train_loader:
                fake_z = netI(x.to(self.device))
                fake_zs.append(fake_z)
        return torch.cat(fake_zs)

    def compute_powers_and_sizes(self, T_train, label):
        powers = []
        for cur_lab in self.present_label:
            if cur_lab != label:
                idxs = torch.where(self.train_gen.targets == cur_lab)[0]
                train_data = torch.utils.data.Subset(self.train_gen, idxs)
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

                netI = self.models[label]['I']
                netI.eval()  # Set the model to evaluation mode
                p_vals = torch.zeros(len(idxs))
                with torch.no_grad():
                    for i, (x, _) in enumerate(train_loader):
                        fake_z = netI(x.to(self.device))
                        T_batch = torch.sqrt(torch.sum(fake_z ** 2, dim=1) + 1)
                        for j in range(len(fake_z)):
                            p = torch.sum(T_train > T_batch[j]) / len(T_train)
                            p_vals[i * self.batch_size + j] = p.item()
                powers.append(np.sum(np.array(p_vals) <= 0.05) / len(idxs))

        sample_sizes = max(powers) - powers + 0.05
        sample_sizes = (sample_sizes / sum(sample_sizes) * len(idxs)).astype(int)
        return sample_sizes



    @staticmethod
    def next_batch(data_iter, train_loader):
        try:
            return next(data_iter)
        except StopIteration:
            # Reset the iterator and return the next batch
            return next(iter(train_loader))

    @staticmethod
    def freeze_batch_norm_layers(net):
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()  # Set the batch norm layer to evaluation mode
                for param in module.parameters():
                    param.requires_grad = False  # Disable gradient updates for batch norm parameters


    def save_model(self, label):
        model_save_file = f'fmnist_param/{self.timestamp}_class{label}.pt'
        torch.save(self.models[label]['I'].state_dict(), model_save_file)

    def validate(self):
        all_p_vals = []
        all_fake_Ts = []

        # Check if T_trains is empty and generate it if necessary
        if self.inference_only:
            for lab in self.present_label:
                idxs = torch.where(torch.Tensor(self.train_gen.targets) == lab)[0]
                train_data = torch.utils.data.Subset(self.train_gen, idxs)
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
                
                # Get fake_zs for further training or analysis
                fake_zs = self.get_fake_zs(lab, train_loader)
                
                # Calculate empirical distribution T_train
                T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
                self.T_trains.append(T_train)

        for lab in self.all_label:
            if torch.is_tensor(self.test_gen.targets):
                idxs = torch.where(self.test_gen.targets == lab)[0]
            else:
                idxs = torch.where(torch.Tensor(self.test_gen.targets) == lab)[0]
            test_data = torch.utils.data.Subset(self.test_gen, idxs)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

            # Initialize p_vals and fake_Ts for the current iteration
            fake_Ts = torch.zeros(len(self.present_label), len(idxs))
            p_vals = torch.zeros(len(self.present_label), len(idxs))

            for pidx, present_label in enumerate(self.present_label):
                T_train = self.T_trains[pidx]
                em_len = len(T_train)
                
                # Use the preloaded "I" model from self.models
                netI = self.models[present_label]['I']

                for i, batch in enumerate(test_loader):
                    images, y = batch
                    x = images.view(-1, self.nc, self.img_size * self.img_size).to(self.device)
                    fake_z = netI(x)
                    T_batch = torch.sqrt(torch.sum(fake_z ** 2, 1) + 1)
                    for j in range(len(fake_z)):
                        p1 = torch.sum(T_train > T_batch[j]) / em_len
                        p = p1
                        fake_Ts[pidx, i * self.batch_size + j] = T_batch[j].item()
                        p_vals[pidx, i * self.batch_size + j] = p.item()

            all_p_vals.append(np.array(p_vals))
            all_fake_Ts.append(np.array(fake_Ts))

        # Visualize the results
        self.visualize_T(all_fake_Ts, classes=self.train_gen.classes)
        self.visualize_p(all_p_vals, classes=self.train_gen.classes)

    def visualize_p(self, all_p_vals, classes):
        # print('-'*100, '\n', ' ' * 45, 'p-values', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label

        if len(present_label) == 1:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 

            for i in range(len(all_label)):
                p_vals_class = all_p_vals[i]
                axs[i].set_xlim([0, 1])
                _ = axs[i].hist(p_vals_class[0, :])
                prop = np.sum(np.array(p_vals_class[0, :] <= 0.05) / len(p_vals_class[0, :]))
                prop = np.round(prop, 4)
                if all_label[i] == present_label[0]:
                    axs[i].set_title('Type I Error: {}'.format(prop), fontsize = 20)
                else:
                    axs[i].set_title('Power: {}'.format(prop), fontsize = 20)
                if i == 0:
                    axs[i].set_ylabel(classes[present_label[0]], fontsize = 25)
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
        fig.supxlabel('Validating', fontsize = 25)
        fig.tight_layout()
        fig.savefig('graphs/size_power.pdf', dpi=150)
        

    def visualize_T(self, all_fake_Cs, classes):
        # print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label
        
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
        fig.supxlabel('Validating', fontsize = 25)
        fig.tight_layout()
        fig.savefig('graphs/fake_T.pdf', dpi=150)

    def save_loss_plots(self, label, GI_losses, MMD_losses, D_losses, GP_losses, Power_losses):
        """Save the losses for the training process to the graphs folder."""
        # Ensure the graphs directory exists
        os.makedirs('graphs', exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(GI_losses, label='GI Loss')
        plt.plot(MMD_losses, label='MMD Loss')
        plt.plot(D_losses, label='D Loss')
        plt.plot(GP_losses, label='GP Loss')
        plt.plot(Power_losses, label='Power Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training Losses for Label {label}')
        plt.legend()
        # Save the plot instead of showing it
        plt.savefig(f'graphs/losses_class{label}.png')
        plt.close()