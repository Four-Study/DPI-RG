import time
import seaborn as sns
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .base_dpi import BaseDPI
from .losses import *
from .mnist import I_MNIST, G_MNIST, f_MNIST, D_MNIST
from .dataloader import get_data_loader


class DPI_ALL(BaseDPI):
    def __init__(self, *args, lr_D, lambda_d, epochs, **kwargs):
        self.epochs = epochs
        self.T_train = None  
        self.lr_D = lr_D
        self.lambda_d = lambda_d
        super().__init__(*args, **kwargs)
        self.z_dim = len(self.present_label)


    def setup_models(self):
        """Setup a single set of models for all classes."""
        if self.validation_only:
            print("Setting up model from existing file.")
            self.load_inverse_model()  # Load single "I" model for validation
        else:
            print("Setting up new models.")
            self.initialize_model()  # Initialize single set of "I", "G", "f" for training

    def initialize_model(self):
        """Initialize a single set of 'I', 'G', 'f', 'D' models for training."""
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netG = G_MNIST(nz=self.z_dim).to(self.device)
        netf = f_MNIST(nz=self.z_dim).to(self.device)
        netD = D_MNIST().to(self.device)
        netI, netG, netf, netD = nn.DataParallel(netI), nn.DataParallel(netG), nn.DataParallel(netf), nn.DataParallel(netD)

        self.models = {'I': netI, 'G': netG, 'f': netf, 'D': netD}
        self.optimizers = {
            'I': optim.Adam(netI.parameters(), lr=self.lr_I, betas=(0.5, 0.999)),
            'G': optim.Adam(netG.parameters(), lr=self.lr_G, betas=(0.5, 0.999)),
            'f': optim.Adam(netf.parameters(), lr=self.lr_f, betas=(0.5, 0.999), weight_decay=self.weight_decay),
            'D': optim.Adam(netD.parameters(), lr=self.lr_D, betas=(0.5, 0.999)),
        }

    def load_inverse_model(self):
        """Load the pre-trained 'I' model for validation."""
        model_save_file = f'{self.params_folder}/model.pt'
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

        print(f"{'-'*50}\nStart training\n{'-'*50}")
        start_time = time.time()

        self.fixed_noise = self.generate_fixed_noise()

        train_loader = get_data_loader(self.train_gen, self.present_label, self.batch_size)

        # Retrieve models and optimizers
        netI, netG, netf, netD = self.models['I'], self.models['G'], self.models['f'], self.models['D']
        optim_I, optim_G, optim_f, optim_D = self.optimizers['I'], self.optimizers['G'], self.optimizers['f'], self.optimizers['D']
        
        # Initialize lists to store losses
        GI_losses, MMD_losses, f_losses, D_losses, GP_losses = [], [], [], [], []

        self.train_epoch(
            netI, netG, netf, netD,
            optim_I, optim_G, optim_f, optim_D,
            train_loader, 
            sample_sizes=None, 
            GI_losses=GI_losses, MMD_losses=MMD_losses,
            f_losses=f_losses, D_losses=D_losses, GP_losses=GP_losses
        )

        # Save the trained model
        self.save_model()

        # Get fake_zs for further training or analysis
        fake_zs = self.get_fake_zs(train_loader)

        # Calculate empirical distribution T_train
        T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
        self.T_train = T_train

        # Save the loss plots to the graphs folder
        self.save_loss_plots(GI_losses, MMD_losses, f_losses, D_losses, GP_losses)

        # Visualize T_trains distribution after training
        self.visualize_T_train()

        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"{'-'*50}")
        print(f"Finish training")
        print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        print(f"{'-'*50}")

    def train_epoch(self, netI, netG, netf, netD,
                    optim_I, optim_G, optim_f, optim_D, train_loader, 
            sample_sizes=None, sampled_idxs=None, GI_losses=[], MMD_losses=[],
                f_losses=[], D_losses = [], GP_losses=[]):

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
            scheduler_f = StepLR(optim_f, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_D = StepLR(optim_D, step_size=self.decay_epochs, gamma=self.gamma)

        # Training started
        for epoch in range(1, self.epochs + 1):
            if (epoch - 1) % max(self.epochs // 4, 1) == 0 or epoch == self.epochs: # 4 means display the images 4 times
                print(f'Epoch = {epoch}')
                self.display_fake_images(netG, epoch)

            data = iter(train_loader)
            
            # 1. Update G, I network
            for p in netf.parameters():
                p.requires_grad = False
            for p in netD.parameters():
                p.requires_grad = False
            for p in netI.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = True
                
            for _ in range(self.critic_iter):
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device) * self.eta
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                netI.zero_grad()
                netG.zero_grad()
                cost_GI = GI_loss(netI, netG, netf, z, fake_z)
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device) * self.eta
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                fake_x = netG(z)
                mmd = mmd_penalty(fake_z, z, kernel="RBF")
                # primal_cost = cost_GI + self.lambda_mmd * mmd
                loss_D = -torch.mean(torch.log(netD(fake_x)))
                primal_cost = cost_GI + self.lambda_d * loss_D
                primal_cost.backward()
                optim_I.step()
                optim_G.step()

            GI_losses.append(cost_GI.cpu().item())
            MMD_losses.append(self.lambda_mmd * mmd.cpu().item())
            
            # 2. Update f, D network
            for p in netf.parameters():
                p.requires_grad = True
            for p in netD.parameters():
                p.requires_grad = True
            for p in netI.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False
                
            for _ in range(self.critic_iter_f):
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device) * self.eta
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                fake_z = netI(x)
                fake_x = netG(z)
                netf.zero_grad()
                netD.zero_grad()
                cost_f = f_loss(netI, netG, netf, z, fake_z)
                loss_D = -torch.mean(torch.log(1-netD(fake_x)))-torch.mean(torch.log(netD(x)))
                images, y = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device) * self.eta
                z = torch.randn(len(images), self.z_dim, device=self.device) * self.std + y_one_hot
                gp_f = gradient_penalty_dual(x.data, z.data, netf, netG, netI)
                dual_cost = cost_f + self.lambda_gp * gp_f + self.lambda_d * loss_D
                dual_cost.backward()
                optim_f.step()
                optim_D.step()
                
            f_losses.append(cost_f.cpu().item())
            D_losses.append(loss_D.cpu().item())
            GP_losses.append(self.lambda_gp * gp_f.cpu().item())
            
            if isinstance(self.decay_epochs, int):
                scheduler_I.step()
                scheduler_G.step()
                scheduler_f.step()
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
                y_one_hot = F.one_hot(y, num_classes=len(self.present_label)).float().to(self.device) * self.eta
                # Subtract the one-hot encoded y from fake_z
                adjusted_fake_z = fake_z - y_one_hot
                adjusted_fake_zs.append(adjusted_fake_z)
        return torch.cat(adjusted_fake_zs)

    def save_model(self):
        model_save_file = f'{self.params_folder}/model.pt'
        torch.save(self.models['I'].state_dict(), model_save_file)

    def validate(self):
        all_p_vals = {label: {} for label in self.all_label}
        all_fake_Ts = {label: {} for label in self.all_label}

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
        netI.eval()  # Set to evaluation mode

        # Create a single test loader for all classes
        test_loader = DataLoader(self.test_gen, batch_size=1, shuffle=True)

        fake_Ts = {label: [] for label in self.all_label}
        p_vals = {label: [] for label in self.all_label}

        for images, y in test_loader:
            x = images.view(-1, self.nc, self.img_size, self.img_size).to(self.device)
            with torch.no_grad():
                fake_z = netI(x)

            for label in self.present_label:
                # Create one-hot encoded y for the current label
                y_one_hot = torch.nn.functional.one_hot(torch.tensor([label] * len(y)), num_classes=len(self.present_label)).float().to(self.device) * self.eta
                # Subtract the one-hot encoded y from fake_z
                adjusted_fake_z = fake_z - y_one_hot
                
                T_batch = torch.sqrt(torch.sum(adjusted_fake_z ** 2, 1) + 1)
                p = torch.tensor([torch.sum(T_train > t) / em_len for t in T_batch])

                # Store results for each true label
                for true_label in self.all_label:
                    mask = (y == true_label)
                    if mask.any():
                        fake_Ts[true_label].extend(T_batch[mask].cpu().tolist())
                        p_vals[true_label].extend(p[mask].cpu().tolist())

        # Convert lists to numpy arrays and store in the main dictionaries
        for true_label in self.all_label:
            all_fake_Ts[true_label] = {label: np.array(fake_Ts[true_label]) for label in self.present_label}
            all_p_vals[true_label] = {label: np.array(p_vals[true_label]) for label in self.present_label}

        # Visualize the results
        self.visualize_T(all_fake_Ts, classes=self.test_gen.classes, path=f'{self.graphs_folder}/fake_T.png')
        self.visualize_p(all_p_vals, classes=self.test_gen.classes, path=f'{self.graphs_folder}/size_power.png')

        print('Finish validation.')

    def visualize_T_train(self):
        """Visualize the distribution of T_train."""
        plt.figure(figsize=(10, 6))
        plt.title('Distribution of T_train', fontsize=16)

        T_train = self.T_train.cpu().numpy()
        sns.kdeplot(T_train, fill=True)

        plt.xlabel('T values')
        plt.ylabel('Density')
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{self.graphs_folder}/T_train.png')
        plt.close()

    def save_loss_plots(self, GI_losses, MMD_losses, f_losses, D_losses, GP_losses):
        """Save the losses for the training process to the graphs folder."""

        plt.figure(figsize=(10, 6))
        
        # Plot with high contrast colors and different line styles without markers
        epochs = range(1, len(GI_losses) + 1)
        plt.plot(epochs, GI_losses, label='GI Loss', color='black', linestyle='-')    # Black, solid line
        # plt.plot(epochs, MMD_losses, label='MMD loss', color='blue', linestyle='--')  # Blue, dashed line
        plt.plot(epochs, f_losses, label='f loss', color='green', linestyle='-.')     # Green, dash-dot line
        plt.plot(epochs, f_losses, label='D loss', color='blue', linestyle='--')     # Green, dash-dot line

        # Combine all losses into one array
        all_losses = np.concatenate([GI_losses, MMD_losses, f_losses, D_losses, GP_losses])
        # Calculate the upper quantile 
        uq = np.percentile(all_losses, 99.5)
        min_loss = np.min(all_losses)

        # Set y-axis limits
        plt.ylim(min_loss - 0.01, uq)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Losses')
        plt.legend()
        
        # Save the plot instead of showing it
        plt.savefig(f'{self.graphs_folder}/losses.png')
        plt.close()

    def generate_fixed_noise(self):
        nclass = len(self.present_label)
        fixed_noise = torch.randn(4 * nclass, self.z_dim, device=self.device) * self.std
        add = F.one_hot(torch.arange(nclass, device=self.device).repeat(4), nclass)
        fixed_noise += add * self.eta
        return fixed_noise

    def display_fake_images(self, netG, epoch):
        nclass = len(self.present_label)
        with torch.no_grad():
            fake = netG(self.fixed_noise.view(4 * nclass, self.z_dim)).view(-1, 1, 28, 28)
        
        plt.figure(figsize=(nclass, 4))
        plt.axis("off")
        plt.title(f'Fake Images for Epoch {epoch}')
        plt.imshow(np.transpose(vutils.make_grid(fake.cpu(), nrow=nclass, padding=2, normalize=True), (1, 2, 0)))
        plt.show()