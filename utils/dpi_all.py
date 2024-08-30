import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .losses import *
from .mnist import I_MNIST, G_MNIST, D_MNIST
from .dataloader import get_dataset, get_data_loader


class DPI:
    def __init__(self, dataset_name, lr_G, lr_I, lr_D, weight_decay, batch_size, epochs, lambda_mmd, lambda_gp, eta, std,
                 present_label, missing_label = [], img_size=28, nc=1, critic_iter=10, critic_iter_d=10, decay_epochs=None, gamma=0.2, device=None, timestamp=None):
        
        self.lr_I = lr_I
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        # self.epochs2 = epochs2
        self.lambda_mmd = lambda_mmd
        self.lambda_gp = lambda_gp
        # self.lambda_power = lambda_power
        self.eta = eta
        self.std = std
        self.img_size = img_size
        self.nc = nc
        self.critic_iter = critic_iter
        self.critic_iter_d = critic_iter_d
        # self.critic_iter_p = critic_iter_p
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.dataset_name = dataset_name
        self.train_gen = get_dataset(dataset_name, train=True)
        self.test_gen = get_dataset(dataset_name, train=False)
        self.present_label = present_label
        self.z_dim = len(present_label)
        self.missing_label = missing_label
        self.all_label = self.present_label + self.missing_label
        self.T_train = None  

        # Set timestamp and mode
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            self.validation_only = False  # Indicates we are in training mode
        else:
            self.timestamp = timestamp
            self.validation_only = True  # Indicates we are in validation mode

        # Initialize or load models based on the mode
        self.setup_models()

    def setup_models(self):
        """Setup a single set of models for all classes."""
        if self.validation_only:
            print("Setting up model from existing file.")
            self.load_inverse_model()  # Load single "I" model for validation
        else:
            print("Setting up new models.")
            self.initialize_model()  # Initialize single set of "I", "G", "D" for training

    def initialize_model(self):
        """Initialize a single set of 'I', 'G', 'D' models for training."""
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netG = G_MNIST(nz=self.z_dim).to(self.device)
        netD = D_MNIST(nz=self.z_dim).to(self.device)
        netI, netG, netD = nn.DataParallel(netI), nn.DataParallel(netG), nn.DataParallel(netD)

        self.models = {'I': netI, 'G': netG, 'D': netD}
        self.optimizers = {
            'I': optim.Adam(netI.parameters(), lr=self.lr_I, betas=(0.5, 0.999)),
            'G': optim.Adam(netG.parameters(), lr=self.lr_G, betas=(0.5, 0.999)),
            'D': optim.Adam(netD.parameters(), lr=self.lr_D, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        }

    def load_inverse_model(self):
        """Load the pre-trained 'I' model for validation."""
        model_save_file = f'fmnist_param/{self.timestamp}_model.pt'
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netI = nn.DataParallel(netI)
        try:
            netI.load_state_dict(torch.load(model_save_file))
            self.models = {'I': netI}
            print(f"Successfully loaded model from {model_save_file}.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model at {model_save_file}. Error: {e}")

    def train(self):
        """Train a single set of models for all present labels."""
        if self.validation_only:
            print("Training is not allowed when a timestamp is provided. Exiting the train method.")
            return

        print(f"{'-'*100}\nStart training\n{'-'*100}")
        start_time = time.time()

        train_loader = get_data_loader(self.train_gen, self.present_label, self.batch_size)

        # Retrieve models and optimizers
        netI, netG, netD = self.models['I'], self.models['G'], self.models['D']
        optim_I, optim_G, optim_D = self.optimizers['I'], self.optimizers['G'], self.optimizers['D']
        
        # Initialize lists to store losses
        GI_losses, MMD_losses, D_losses, GP_losses = [], [], [], []

        self.train_epoch(
            netI, netG, netD, optim_I, optim_G, optim_D,
            train_loader, 
            sample_sizes=None, 
            GI_losses=GI_losses, MMD_losses=MMD_losses,
            D_losses=D_losses, GP_losses=GP_losses
        )

        # Save the trained model
        self.save_model()

        # Get fake_zs for further training or analysis
        fake_zs = self.get_fake_zs(train_loader)

        # Calculate empirical distribution T_train
        T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
        self.T_train = T_train

        # Save the loss plots to the graphs folder
        self.save_loss_plots(GI_losses, MMD_losses, D_losses, GP_losses)

        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"{'-'*100}")
        print(f"Finish training")
        print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        print(f"{'-'*100}")

    def train_epoch(self, netI, netG, netD, optim_I, optim_G, optim_D, train_loader, 
            sample_sizes=None, sampled_idxs=None, GI_losses=[], MMD_losses=[],
                D_losses=[], GP_losses=[]):

        if sampled_idxs is None:
            sampled_idxs = {}
            for cur_lab in self.present_label:
                if torch.is_tensor(self.train_gen.targets):
                    temp = torch.where(self.train_gen.targets == cur_lab)[0] 
                else:
                    temp = torch.where(torch.Tensor(self.train_gen.targets) == cur_lab)[0] 
                sampled_idxs[cur_lab] = temp
                
        if sample_sizes is None:
            sample_sizes = {cur_lab: int(len(train_loader.dataset.indices) / len(self.present_label)) for cur_lab in self.present_label}
            
        # Learning rate schedulers
        if isinstance(self.decay_epochs, int):
            scheduler_I = StepLR(optim_I, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_G = StepLR(optim_G, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_D = StepLR(optim_D, step_size=self.decay_epochs, gamma=self.gamma)

        # Training for this label started
        for epoch in range(1, self.epochs):
            if (epoch - 1) % max(self.epochs // 4, 1) == 0 or epoch == self.epochs:
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
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device)
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                # fake_x = netG(z)
                netI.zero_grad()
                netG.zero_grad()
                cost_GI = GI_loss(netI, netG, netD, z, fake_z)
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device)
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                mmd = mmd_penalty(fake_z, z, kernel="RBF")
                primal_cost = cost_GI + self.lambda_mmd * mmd
                primal_cost.backward()
                optim_I.step()
                optim_G.step()

            GI_losses.append(cost_GI.cpu().item())
            MMD_losses.append(self.lambda_mmd * mmd.cpu().item())
            
            # 2. Update D network
            for p in netD.parameters():
                p.requires_grad = True
            for p in netI.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False
                
            for _ in range(self.critic_iter_d):
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device)
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                netD.zero_grad()
                cost_D = D_loss(netI, netG, netD, z, fake_z)
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device)
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                gp_D = gradient_penalty_dual(x.data, z.data, netD, netG, netI)
                dual_cost = cost_D + self.lambda_gp * gp_D
                dual_cost.backward()
                optim_D.step()
                
            D_losses.append(cost_D.cpu().item())
            GP_losses.append(self.lambda_gp * gp_D.cpu().item())
            
            if isinstance(self.decay_epochs, int):
                scheduler_I.step()
                scheduler_G.step()
                scheduler_D.step()

    def get_fake_zs(self, train_loader):
        netI = self.models['I']
        netI.eval()  # Set the model to evaluation mode
        adjusted_fake_zs = []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(self.device)
                fake_z = netI(x)
                # Create one-hot encoded y
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device)
                # Subtract the one-hot encoded y from fake_z
                adjusted_fake_z = fake_z - y_one_hot
                adjusted_fake_zs.append(adjusted_fake_z)
        return torch.cat(adjusted_fake_zs)

    def compute_powers_and_sizes(self, T_train):
        powers = {}
        for cur_lab in self.present_label:
            idxs = torch.where(self.train_gen.targets == cur_lab)[0]
            train_data = torch.utils.data.Subset(self.train_gen, idxs)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

            netI = self.models['I']
            netI.eval()  # Set the model to evaluation mode
            p_vals = torch.zeros(len(idxs))
            with torch.no_grad():
                for i, (x, _) in enumerate(train_loader):
                    fake_z = netI(x.to(self.device))
                    T_batch = torch.sqrt(torch.sum(fake_z ** 2, dim=1) + 1)
                    for j in range(len(fake_z)):
                        p = torch.sum(T_train > T_batch[j]) / len(T_train)
                        p_vals[i * self.batch_size + j] = p.item()
            powers[cur_lab] = np.sum(np.array(p_vals.numpy()) <= 0.05) / len(idxs)

        sample_sizes = {lab: max(powers.values()) - pow + 0.05 for lab, pow in powers.items()}
        total = sum(sample_sizes.values())
        sample_sizes = {lab: int((size / total) * len(idxs)) for lab, size in sample_sizes.items()}

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

    def save_model(self):
        model_save_file = f'fmnist_param/{self.timestamp}_model.pt'
        torch.save(self.models['I'].state_dict(), model_save_file)

    def validate(self):
        all_p_vals = {}
        all_fake_Ts = {}

        # Check if T_train is empty and generate it if necessary
        if self.validation_only:
            idxs = torch.where(torch.isin(torch.Tensor(self.train_gen.targets), torch.tensor(self.present_label)))[0]
            train_data = torch.utils.data.Subset(self.train_gen, idxs)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
            
            # Get fake_zs for further training or analysis
            fake_zs = self.get_fake_zs(train_loader)
            
            # Calculate empirical distribution T_train
            T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
            self.T_train = T_train
        else:
            T_train = self.T_train
        em_len = len(T_train)
        # Use the single preloaded "I" model
        netI = self.models['I']
        netI.train()

        for lab in self.all_label:
            if torch.is_tensor(self.test_gen.targets):
                idxs = torch.where(self.test_gen.targets == lab)[0]
            else:
                idxs = torch.where(torch.Tensor(self.test_gen.targets) == lab)[0]
            test_data = torch.utils.data.Subset(self.test_gen, idxs)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

            # Initialize p_vals and fake_Ts for the current iteration
            fake_Ts = {label: torch.zeros(len(idxs)) for label in self.present_label}
            p_vals = {label: torch.zeros(len(idxs)) for label in self.present_label}

            for i, (images, y) in enumerate(test_loader):
                x = images.view(-1, self.nc, self.img_size, self.img_size).to(self.device)
                with torch.no_grad():
                    fake_z = netI(x)

                for label in self.present_label:
                    # Create one-hot encoded y for the current label
                    y_one_hot = torch.nn.functional.one_hot(torch.tensor([label] * len(y)), num_classes=len(self.present_label)).float().to(self.device)
                    # Subtract the one-hot encoded y from fake_z
                    adjusted_fake_z = fake_z - y_one_hot
                    
                    T_batch = torch.sqrt(torch.sum(adjusted_fake_z ** 2, 1) + 1)
                    for j in range(len(fake_z)):
                        p = torch.sum(T_train > T_batch[j]) / em_len
                        fake_Ts[label][i * 1 + j] = T_batch[j].item()
                        p_vals[label][i * 1 + j] = p.item()

            all_p_vals[lab] = {k: v.numpy() for k, v in p_vals.items()}
            all_fake_Ts[lab] = {k: v.numpy() for k, v in fake_Ts.items()}

        # Visualize the results
        self.visualize_T(all_fake_Ts, classes=self.train_gen.classes)
        self.visualize_p(all_p_vals, classes=self.train_gen.classes)

        print('Finish validation.')

    def visualize_p(self, all_p_vals, classes):
        # print('-'*100, '\n', ' ' * 45, 'p-values', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label

        if len(present_label) == 1:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 

            for i, lab in enumerate(all_label):
                p_vals_class = all_p_vals[lab]
                axs[i].set_xlim([0, 1])
                _ = axs[i].hist(p_vals_class[present_label[0]])
                prop = np.sum(np.array(p_vals_class[present_label[0]] <= 0.05) / len(p_vals_class[present_label[0]]))
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

            for i, val_lab in enumerate(all_label):
                p_vals_class = all_p_vals[val_lab]
                for j, train_lab in enumerate(present_label):
                    axs[j, i].set_xlim([0, 1])
                    _ = axs[j, i].hist(p_vals_class[train_lab])
                    prop = np.sum(np.array(p_vals_class[train_lab] <= 0.05) / len(p_vals_class[train_lab]))
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
        fig.supxlabel('Validation', fontsize = 25)
        fig.tight_layout()
        fig.savefig(f'graphs/{self.timestamp}_size_power.png', dpi=150)
        plt.close(fig)
        

    def visualize_T(self, all_fake_Cs, classes):
        # print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label
        
        if len(present_label) == 1:
            fig, axs = plt.subplots(1, len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 
            llim = np.min([np.min(vals[present_label[0]]) for vals in all_fake_Cs.values()])
            rlim = np.quantile(np.concatenate([vals[present_label[0]] for vals in all_fake_Cs.values()]), 0.9)
            for i, lab in enumerate(all_label):
                fake_Cs = all_fake_Cs[lab]
                axs[i].set_ylabel(classes[lab], fontsize = 25)
                axs[i].set_xlim([llim, rlim])
                _ = axs[i].hist(fake_Cs[present_label[0]])
                # axs[i].set_title('Label {} from Label {}\'s net'.format(lab, present_label[0]), fontsize = 20)

                if i == len(all_label) - 1:
                    axs[i].set_xlabel(classes[present_label[0]], fontsize = 25)
        else:
            fig, axs = plt.subplots(len(present_label), len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 
            llim = np.min([np.min(list(vals.values())) for vals in all_fake_Cs.values()])
            rlim = np.quantile(np.concatenate([np.concatenate(list(vals.values())) for vals in all_fake_Cs.values()]), 0.95)
            for i, val_lab in enumerate(all_label):
                fake_Cs = all_fake_Cs[val_lab]
                for j, train_lab in enumerate(present_label):
                    axs[j, i].set_xlim([llim, rlim])
                    _ = axs[j, i].hist(fake_Cs[train_lab])
                    # axs[j, i].set_title('Label {} from Label {}\'s net'.format(val_lab, train_lab))
                    if i == 0:
                        axs[j, i].set_ylabel(classes[train_lab], fontsize = 25)
                    if j == len(present_label) - 1:
                        axs[j, i].set_xlabel(classes[val_lab], fontsize = 25)
        
        fig.supylabel('Training', fontsize = 25)
        fig.supxlabel('Validation', fontsize = 25)
        fig.tight_layout()
        fig.savefig(f'graphs/{self.timestamp}_fake_T.png', dpi=150)
        plt.close(fig)

    def save_loss_plots(self, GI_losses, MMD_losses, D_losses, GP_losses):
        """Save the losses for the training process to the graphs folder."""
        # Ensure the graphs directory exists
        os.makedirs('graphs', exist_ok=True)

        plt.figure(figsize=(10, 6))
        
        # Plot with high contrast colors and different line styles without markers
        epochs = range(1, len(GI_losses) + 1)
        plt.plot(epochs, GI_losses, label='GI Loss', color='black', linestyle='-')    # Black, solid line
        plt.plot(epochs, MMD_losses, label='MMD Loss', color='blue', linestyle='--')  # Blue, dashed line
        plt.plot(epochs, D_losses, label='D Loss', color='green', linestyle='-.')     # Green, dash-dot line
        # plt.plot(epochs, GP_losses, label='GP Loss', color='red', linestyle=':')      # Red, dotted line
        # plt.plot(epochs, Power_losses, label='Power Loss', color='purple', linestyle=':') # Purple, solid line
        
        # Combine all losses into one array
        all_losses = np.concatenate([GI_losses, MMD_losses, D_losses, GP_losses])
        # Calculate the 99th percentile
        uq = np.percentile(all_losses, 99)
        min_loss = np.min(all_losses)

        # Set y-axis limits
        plt.ylim(min_loss - 0.01, uq)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Losses')
        plt.legend()
        
        # Save the plot instead of showing it
        plt.savefig(f'graphs/{self.timestamp}_losses.png')
        plt.close()