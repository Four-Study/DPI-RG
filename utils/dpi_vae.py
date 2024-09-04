import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

class DPI_VAE:
    def __init__(self, nc, ngf, ndf, nclass, lr, device):
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.nclass = nclass
        self.const = nclass ** 0.5 + 3
        self.device = device

        self.model = ConvVAE(nclass, nc, ngf, ndf).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 10))

        # Generate fixed noise for display
        self.fixed_noise = self.generate_fixed_noise()

    def generate_fixed_noise(self):
        fixed_noise = torch.randn(4 * self.nclass, self.nclass, device=self.device)
        add = F.one_hot(torch.arange(self.nclass, device=self.device).repeat(4), self.nclass)
        fixed_noise += add * self.const
        return fixed_noise

    def train(self, train_loader):
        self.model.train()
        losses = []

        for batch_idx, (real_X, y) in enumerate(train_loader):
            real_X, y = real_X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(real_X)
            real_X = (real_X.float() + 1) / 2
            loss = self.loss_function(x_recon, real_X, mu, logvar, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        self.lr_scheduler.step()
        return np.mean(losses)

    def loss_function(self, x_recon, x, mu, logvar, y):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        mu_q = F.one_hot(y, self.nclass) * self.const
        kl_divergence = -0.5 * torch.sum(1 + logvar - (mu - mu_q).pow(2) - logvar.exp())
        return recon_loss + kl_divergence * 5

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def reparametrize(self, mu, logvar):
        return self.model.reparametrize(mu, logvar)

    def display_fake_images(self):
        self.model.eval()
        with torch.no_grad():
            fake = self.decode(self.fixed_noise.view(4 * self.nclass, self.nclass)).view(-1, 1, 28, 28)
        
        plt.figure(figsize=(self.nclass, 4))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(vutils.make_grid(fake.cpu(), nrow=self.nclass, padding=2, normalize=True), (1, 2, 0)))
        plt.show()

    def compute_empiricals(self, trainset_A, batch_size):
        self.model.eval()
        empiricals = []
        for lab in range(self.nclass):
            # Get the fake_Bs for each label
            idxs2 = torch.where(trainset_A.targets == lab)[0] 
            train_data2 = Subset(trainset_A, idxs2)
            train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True)
            fake_Bs = []
            with torch.no_grad(): 
                for i, batch in enumerate(train_loader2):
                    real_X, _ = batch
                    real_X = real_X.to(self.device)
                    mu, logvar = self.model.encode(real_X)
                    fake_B = torch.randn_like(mu) + mu
                    fake_Bs.append(fake_B)
            fake_Bs = torch.cat(fake_Bs)
            # Get the empirical distribution for each label
            fake_Bs -= F.one_hot(torch.Tensor([lab] * len(idxs2)).to(torch.int64), self.nclass).to(self.device) * self.const
            empirical = torch.sum(torch.square(fake_Bs), 1) 
            empiricals.append(empirical)
        return empiricals

    def validate(self, testset_A, batch_size, present_label, all_label, empiricals, side='one-sided'):
        self.model.eval()
        nclass = len(present_label)
        p_vals_classes = []
        all_fake_Cs = []
        
        for lab in all_label:    
            if torch.is_tensor(testset_A.targets):
                idxs_2 = torch.where(testset_A.targets == lab)[0] 
            else:
                idxs_2 = torch.where(torch.Tensor(testset_A.targets) == lab)[0] 
            test_data2 = Subset(testset_A, idxs_2)
            test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=False)

            p_vals_class = torch.zeros(len(present_label), len(idxs_2)) 
            fake_Cs = torch.zeros(len(present_label), len(idxs_2))

            for pidx in range(len(present_label)):
                empirical = empiricals[pidx]
                em_len = len(empiricals[pidx])
                for i, batch in enumerate(test_loader2):
                    real_A, _ = batch
                    real_A = real_A.to(self.device)
                    bs = len(real_A)
                    mu, logvar = self.model.encode(real_A)
                    fake_B = torch.randn_like(mu) + mu
                    fake_B -= F.one_hot(torch.tensor([present_label[pidx]] * bs, device=self.device, dtype=torch.int64), nclass) * self.const
                    fake_C = torch.sum(torch.square(fake_B), 1)
                        
                    for j in range(len(fake_C)):
                        if side == 'two-sided':
                            p1 = torch.sum(fake_C[j] > empirical) / em_len
                            p2 = torch.sum(fake_C[j] < empirical) / em_len
                            p = 2 * torch.min(p1, p2)
                        elif side == 'one-sided':
                            p = torch.sum(fake_C[j] < empirical) / em_len
                        p_vals_class[pidx, i * batch_size + j] = p.item()
                        fake_Cs[pidx, i * batch_size + j] = fake_C[j].item()

            p_vals_classes.append(np.array(p_vals_class))
            all_fake_Cs.append(np.array(fake_Cs))

        return p_vals_classes, all_fake_Cs


    def visualize_p(self, all_p_vals, present_label, all_label, classes):
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
        fig.savefig(f'{self.graphs_folder}/{self.timestamp}_size_power.png', dpi=150)
        plt.close(fig)
        

    def visualize_T(self, all_fake_Cs, present_label, all_label, classes):
        # print('-'*100, '\n', ' ' * 45, 'fake numbers', '\n', '-'*100, sep = '')
        present_label = self.present_label
        all_label = self.all_label
        
        if len(present_label) == 1:
            fig, axs = plt.subplots(1, len(all_label), 
                                    figsize=(5*len(all_label), 5*len(present_label)))

            matplotlib.rc('xtick', labelsize=15) 
            matplotlib.rc('ytick', labelsize=15) 
            llim = np.min([np.min(vals[present_label[0]]) for vals in all_fake_Cs.values()])
            rlim = np.quantile(np.concatenate([vals[present_label[0]] for vals in all_fake_Cs.values()]), 0.98)
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
        fig.savefig(f'{self.graphs_folder}/{self.timestamp}_fake_T.png', dpi=150)
        plt.close(fig)

class ConvVAE(nn.Module):
    def __init__(self, nz, nc, ngf, ndf):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    
