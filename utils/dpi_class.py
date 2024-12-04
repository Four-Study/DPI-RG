import time
import seaborn as sns
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .base_dpi import BaseDPI
from .losses import *
from .mnist import I_MNIST, G_MNIST, f_MNIST
from .dataloader import get_data_loader


class DPI_CLASS(BaseDPI):
    def __init__(self, *args, z_dim, lambda_power, critic_iter_p, epochs1, epochs2, **kwargs):
        
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.critic_iter_p = critic_iter_p
        self.T_trains = {}  
        self.z_dim = z_dim
        self.lambda_power = lambda_power
        super().__init__(*args, **kwargs)

    def setup_models(self):
        """Setup models based on exitsting files or not."""
        if self.validation_only:
            print("Setting up models from existing files.")
            for label in self.present_label:
                self.load_inverse_model(label)  # Load "I" model for validation
        else:
            print("Setting up new models.")
            for label in self.present_label:
                self.initialize_model(label)  # Initialize "I", "G", "f" for training

    def initialize_model(self, label):
        """Initialize 'I', 'G', 'f' models for a given label for training."""
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netG = G_MNIST(nz=self.z_dim).to(self.device)
        netf = f_MNIST(nz=self.z_dim).to(self.device)
        netI, netG, netf = nn.DataParallel(netI), nn.DataParallel(netG), nn.DataParallel(netf)

        self.models[label] = {'I': netI, 'G': netG, 'f': netf}
        self.optimizers[label] = {
            'I': optim.Adam(netI.parameters(), lr=self.lr_I, betas=(0.5, 0.999)),
            'G': optim.Adam(netG.parameters(), lr=self.lr_G, betas=(0.5, 0.999)),
            'f': optim.Adam(netf.parameters(), lr=self.lr_f, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        }

    def load_inverse_model(self, label):
        """Load the pre-trained 'I' model for validation."""
        model_save_file = f'{self.params_folder}/{self.timestamp}_class{label}.pt'
        netI = I_MNIST(nz=self.z_dim).to(self.device)
        netI = nn.DataParallel(netI)
        try:
            netI.load_state_dict(torch.load(model_save_file, weights_only=True))
            self.models[label] = {'I': netI}
            print(f"Successfully loaded model for label {label} from {model_save_file}.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model for label {label} at {model_save_file}. Error: {e}")


    def train(self):
        """Train models for each present label."""
        if self.validation_only:
            print("Training is not allowed when a timestamp is provided. Exiting the train method.")
            return
        
        print(f"{'-'*50}\nStart training\n{'-'*50}")    
        start_time = time.time()
        
        for label in self.present_label:
            print(f"{'-'*50}\nStart to train label: {label}\n{'-'*50}")

            self.fixed_noise = self.generate_fixed_noise()

            train_loader = get_data_loader(self.train_gen, [label], self.batch_size)

            # Retrieve models and optimizers for the current label
            netI, netG, netf = self.models[label]['I'], self.models[label]['G'], self.models[label]['f']
            optim_I, optim_G, optim_f = self.optimizers[label]['I'], self.optimizers[label]['G'], self.optimizers[label]['f']
            
            # Initialize lists to store losses
            GI_losses, MMD_losses, f_losses, GP_losses, Power_losses = [], [], [], [], []

            # First round of training
            self.train_label(
                label, netI, netG, netf, optim_I, optim_G, optim_f,
                train_loader, 0, self.epochs1,
                sample_sizes=None, 
                GI_losses=GI_losses, MMD_losses=MMD_losses,
                f_losses=f_losses, GP_losses=GP_losses, Power_losses=Power_losses
            )
            
            if self.epochs2 - self.epochs1 > 0:
                # Get fake_zs for further training or analysis
                fake_zs = self.get_fake_zs(label, train_loader)

                # Calculate empirical distribution T_train
                T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)

                # Compute powers and new sample sizes
                sample_sizes = self.compute_sample_sizes(T_train, label)

                # Freeze batch normalization layers before the second round of training
                # self.freeze_batch_norm_layers(netI)

                # Second round of training with new sample sizes
                self.train_label(
                    label, netI, netG, netf, optim_I, optim_G, optim_f,
                    train_loader, self.epochs1, self.epochs2,
                    sample_sizes=sample_sizes, 
                    GI_losses=GI_losses, MMD_losses=MMD_losses,
                    f_losses=f_losses, GP_losses=GP_losses, Power_losses=Power_losses
                )

            # Save the trained model
            self.save_model(label)

            # Get fake_zs for further training or analysis
            fake_zs = self.get_fake_zs(label, train_loader)

            # Calculate empirical distribution T_train
            T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
            self.T_trains[label] = T_train

            # Save the loss plots to the graphs folder
            self.save_loss_plots(label, GI_losses, MMD_losses, f_losses, GP_losses, Power_losses)
        
        # Visualize T_trains distribution after training all labels
        self.visualize_T_trains()

        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{'-'*50}")
        print(f"Finish training")
        print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        print(f"{'-'*50}")

    def train_label(self, label, netI, netG, netf, optim_I, optim_G, optim_f, train_loader, start_epoch, end_epoch, 
            sample_sizes=None, sampled_idxs=None, GI_losses=[], MMD_losses=[],
                f_losses=[], GP_losses=[], Power_losses=[]):

        imbalanced = True
        if sampled_idxs is None:
            imbalanced = False
            sampled_idxs = {}
            for cur_lab in self.present_label:
                if torch.is_tensor(self.train_gen.targets):
                    temp = torch.where(self.train_gen.targets == cur_lab)[0] 
                else:
                    temp = torch.where(torch.Tensor(self.train_gen.targets) == cur_lab)[0] 
                sampled_idxs[cur_lab] = temp
                
        if sample_sizes is None:
            sample_sizes = {cur_lab: int(len(train_loader.dataset.indices) / len(self.present_label)) for cur_lab in self.present_label if cur_lab != label}
            
        # Learning rate schedulers
        if isinstance(self.decay_epochs, int):
            scheduler_I = StepLR(optim_I, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_G = StepLR(optim_G, step_size=self.decay_epochs, gamma=self.gamma)
            scheduler_f = StepLR(optim_f, step_size=self.decay_epochs, gamma=self.gamma)

        # Training for this label started
        for epoch in range(start_epoch + 1, end_epoch + 1):
            if (epoch - 1) % max(self.epochs2 // 4, 1) == 0 or epoch == self.epochs2:
                print(f'Epoch = {epoch}')
                self.display_fake_images(netG)

            data = iter(train_loader)

            # Train in alternative hypothesis
            idxs2 = torch.Tensor([])
            for cur_lab in self.present_label:
                if cur_lab != label:
                    temp = sampled_idxs[cur_lab]
                    idxs2 = torch.cat([idxs2, temp[np.random.choice(len(temp), sample_sizes[cur_lab], replace=imbalanced)]])
            idxs2 = idxs2.int()
            train_data2 = torch.utils.data.Subset(self.train_gen, idxs2)
            train_loader2 = DataLoader(train_data2, batch_size=self.batch_size)
            data2 = iter(train_loader2)
            
            # 3. Update I network
            for p in netf.parameters():
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

            Power_losses.append(loss_power.cpu().item())
            
            # 1. Update G, I network
            for p in netf.parameters():
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
                cost_GI = GI_loss(netI, netG, netf, z, fake_z)
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                fake_z = netI(x)
                mmd = mmd_penalty(fake_z, z, kernel="RBF")
                primal_cost = cost_GI + self.lambda_mmd * mmd
                primal_cost.backward()
                optim_I.step()
                optim_G.step()

            GI_losses.append(cost_GI.cpu().item())
            MMD_losses.append(self.lambda_mmd * mmd.cpu().item())
            
            # 2. Update f network
            for p in netf.parameters():
                p.requires_grad = True
            for p in netI.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False
                
            for _ in range(self.critic_iter_f):
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                fake_z = netI(x)
                netf.zero_grad()
                cost_f = f_loss(netI, netG, netf, z, fake_z)
                images, _ = self.next_batch(data, train_loader)
                x = images.view(len(images), self.nc * self.img_size ** 2).to(self.device)
                z = torch.randn(len(images), self.z_dim).to(self.device)
                gp_f = gradient_penalty_dual(x.data, z.data, netf, netG, netI)
                dual_cost = cost_f + self.lambda_gp * gp_f
                dual_cost.backward()
                optim_f.step()
                
            f_losses.append(cost_f.cpu().item())
            GP_losses.append(self.lambda_gp * gp_f.cpu().item())
            
            if isinstance(self.decay_epochs, int):
                scheduler_I.step()
                scheduler_G.step()
                scheduler_f.step()

    def get_fake_zs(self, label, train_loader):
        netI = self.models[label]['I']
        # netI.eval()  # Set the model to evaluation mode
        fake_zs = []
        with torch.no_grad():
            for x, _ in train_loader:
                fake_z = netI(x.to(self.device))
                fake_zs.append(fake_z)
        return torch.cat(fake_zs)

    def compute_sample_sizes(self, T_train, label):
        powers = {}
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
                powers[cur_lab] = np.sum(np.array(p_vals) <= 0.05) / len(idxs)

        sample_sizes = {lab: max(powers.values()) - pow + 0.05 for lab, pow in powers.items()}
        total = sum(sample_sizes.values())
        sample_sizes = {lab: int((size / total) * len(idxs)) for lab, size in sample_sizes.items()}

        return sample_sizes

    def save_model(self, label):
        model_save_file = f'{self.params_folder}/{self.timestamp}_class{label}.pt'
        torch.save(self.models[label]['I'].state_dict(), model_save_file)

    def validate(self):

        all_p_vals = {label: {} for label in self.all_label}
        all_fake_Ts = {label: {} for label in self.all_label}
    
        # Calculate T_trains for present labels first
        if self.validation_only:
            for label in self.present_label:
                idxs = torch.where(self.train_gen.targets == label)[0]
                train_data = torch.utils.data.Subset(self.train_gen, idxs)
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
                
                # Get fake_zs for further training or analysis
                fake_zs = self.get_fake_zs(label, train_loader)
                
                # Calculate empirical distribution T_train
                T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
                self.T_trains[label] = T_train

        # Create a single test loader for all classes
        test_loader = DataLoader(self.test_gen, batch_size=1, shuffle=False)

        for label in self.present_label:
            T_train = self.T_trains[label]
            em_len = len(T_train)
            
            # Use the preloaded "I" model from self.models
            netI = self.models[label]['I']
            # netI.eval()  # Set to evaluation mode

            fake_Ts = {lab: [] for lab in self.all_label}
            p_vals = {lab: [] for lab in self.all_label}

            for batch in test_loader:
                images, y = batch
                x = images.view(-1, self.nc, self.img_size, self.img_size).to(self.device)
                fake_z = netI(x)
                T_batch = torch.sqrt(torch.sum(fake_z ** 2, 1) + 1)
                
                p = torch.tensor([torch.sum(T_train > t) / em_len for t in T_batch])

                # Store results for each true label
                for true_label in self.all_label:
                    mask = (y == true_label)
                    if mask.any():
                        fake_Ts[true_label].extend(T_batch[mask].cpu().tolist())
                        p_vals[true_label].extend(p[mask].cpu().tolist())

            # Convert lists to numpy arrays and store in the main dictionaries
            for true_label in self.all_label:
                all_fake_Ts[true_label][label] = np.array(fake_Ts[true_label])
                all_p_vals[true_label][label] = np.array(p_vals[true_label])

        # Visualize the results
        self.visualize_T(all_fake_Ts, classes=self.test_gen.classes)
        self.visualize_p(all_p_vals, classes=self.test_gen.classes)

        print('Finish validation.')

    def validate_w_classifier(self, classifier):
        all_p_vals = {label: {} for label in self.all_label}
        all_fake_Ts = {label: {} for label in self.all_label}

        # Calculate T_trains for present labels first
        if self.validation_only:
            for label in self.present_label:
                idxs = torch.where(self.train_gen.targets == label)[0]
                train_data = torch.utils.data.Subset(self.train_gen, idxs)
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
                
                # Get fake_zs for further training or analysis
                fake_zs = self.get_fake_zs(label, train_loader)
                
                # Calculate empirical distribution T_train
                T_train = torch.sqrt(torch.sum(fake_zs ** 2, dim=1) + 1)
                self.T_trains[label] = T_train

        # Set classifier to evaluation mode
        classifier.eval()

        # First, classify all images
        all_images = []
        all_labels = []
        predicted_labels = []
        with torch.no_grad():
            for images, labels in DataLoader(self.test_gen, batch_size=self.batch_size, shuffle=False):
                # Filter for present labels
                mask = torch.tensor([label in self.present_label for label in labels])
                if not mask.any():
                    continue
                
                images = images[mask]
                labels = labels[mask]
                
                x = images.view(images.shape[0], -1).to(self.device)
                outputs = classifier(x)
                _, preds = torch.max(outputs, 1)
                predicted_labels.extend(preds.cpu().numpy())
                all_images.extend(images)
                all_labels.extend(labels)

        # Sort the data based on predicted labels
        sorted_indices = sorted(range(len(predicted_labels)), key=lambda k: predicted_labels[k])
        sorted_images = [all_images[i] for i in sorted_indices]
        sorted_labels = [all_labels[i] for i in sorted_indices]

        # Create grouped_data to replace test_loader
        grouped_data = {lab: {'images': [], 'labels': []} for lab in self.all_label}

        for idx, label in enumerate(sorted_labels):
            grouped_data[label.item()]['images'].append(sorted_images[idx])
            grouped_data[label.item()]['labels'].append(label.item())

        # Process data for each present label
        for label in self.present_label:
            T_train = self.T_trains[label]
            em_len = len(T_train)
            
            # Use the preloaded "I" model from self.models
            netI = self.models[label]['I']
            
            fake_Ts = {lab: [] for lab in self.all_label}
            p_vals = {lab: [] for lab in self.all_label}

            # Iterate over grouped_data instead of test_loader
            for true_label, data in grouped_data.items():
                images = torch.stack(data['images']).to(self.device)
                y = torch.tensor(data['labels']).to(self.device)
                
                # Process the images using your model
                fake_z = netI(images)
                T_batch = torch.sqrt(torch.sum(fake_z ** 2, 1) + 1)
                
                # Compute probabilities (same as before)
                p = torch.tensor([torch.sum(T_train > t) / em_len for t in T_batch])

                # Store results for each true label
                mask = (y == true_label).cpu()
                if mask.any():
                    fake_Ts[true_label].extend(T_batch[mask].cpu().tolist())
                    p_vals[true_label].extend(p[mask].cpu().tolist())

            # Convert lists to numpy arrays and store in the main dictionaries
            for true_label in self.all_label:
                all_fake_Ts[true_label][label] = np.array(fake_Ts[true_label])
                all_p_vals[true_label][label] = np.array(p_vals[true_label])

        # Visualize the results
        self.visualize_T(all_fake_Ts, classes=self.test_gen.classes)
        self.visualize_p(all_p_vals, classes=self.test_gen.classes)

        print('Finish validation with classifier.')
    
    def visualize_T_trains(self):
        """Visualize the distribution of T_trains for all labels in a single row."""
        num_classes = len(self.T_trains)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(3*num_classes, 4))
        fig.suptitle('Distribution of T_trains for all classes', fontsize=16)

        for idx, label in enumerate(self.present_label):
            ax = axes[idx]

            T_train = self.T_trains[label]
            sns.kdeplot(T_train.cpu().numpy(), fill=True, ax=ax)
            ax.set_title(f'Class {label}')
            ax.set_xlabel('T values')
            ax.set_ylabel('Density')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
        
        # Save the plot
        plt.savefig(f'{self.graphs_folder}/{self.timestamp}_T_trains.png')
        plt.close()

    def save_loss_plots(self, label, GI_losses, MMD_losses, f_losses, GP_losses, Power_losses):
        """Save the losses for the training process to the graphs folder."""

        plt.figure(figsize=(10, 6))
        
        # Plot with high contrast colors and different line styles without markers
        epochs = range(1, len(GI_losses) + 1)
        plt.plot(epochs, GI_losses, label='GI Loss', color='black', linestyle='-')    # Black, solid line
        plt.plot(epochs, MMD_losses, label='MMD Loss', color='blue', linestyle='--')  # Blue, dashed line
        plt.plot(epochs, f_losses, label='f loss', color='green', linestyle='-.')     # Green, dash-dot line
        # plt.plot(epochs, GP_losses, label='GP Loss', color='red', linestyle=':')      # Red, dotted line
        plt.plot(epochs, Power_losses, label='Power Loss', color='purple', linestyle=':') # Purple, solid line

        # Add vertical line at self.epochs1
        plt.axvline(x=self.epochs1, color='red', linestyle='-', label='Change of alternative samples')
        
        # Combine all losses into one array
        all_losses = np.concatenate([GI_losses, MMD_losses, f_losses, GP_losses, Power_losses])
        # Calculate the upper quantile 
        uq = np.percentile(all_losses, 99.5)
        min_loss = np.min(all_losses)

        # Set y-axis limits
        plt.ylim(min_loss - 0.01, uq)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Losses for Class {label}')
        plt.legend()
        
        # Save the plot instead of showing it
        plt.savefig(f'{self.graphs_folder}/{self.timestamp}_losses_class{label}.png')
        plt.close()

    def generate_fixed_noise(self):
        fixed_noise = torch.randn(2 * 10, self.z_dim, device=self.device) * self.std
        return fixed_noise

    def display_fake_images(self, netG):
        with torch.no_grad():
            fake = netG(self.fixed_noise.view(2 * 10, self.z_dim)).view(-1, 1, 28, 28)
        
        plt.figure(figsize=(10, 2))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(vutils.make_grid(fake.cpu(), nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.show()