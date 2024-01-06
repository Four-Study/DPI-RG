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
from torch.utils.data import DataLoader
from utils.losses import *

random.seed(1)
np.random.seed(1)
# matplotlib.use('Agg')

# ****************************
# *** Mixture of Gaussians ***
# ****************************
# Data simulator for mixture of gaussians example
# def inf_train_gen(DATASET, BATCH_SIZE=1024, std=0.02):
#     if DATASET == "8gaussians":
#         n = 8
#         radius = 2.
#         delta_theta = 2*np.pi/n
#         centers_x = []
#         centers_y = []
#         for i in range(n):
#             centers_x.append(radius * np.cos(i * delta_theta))
#             centers_y.append(radius * np.sin(i * delta_theta))
#         centers_x = np.expand_dims(np.array(centers_x), 1)
#         centers_y = np.expand_dims(np.array(centers_y), 1)
#         centers = np.concatenate((centers_x, centers_y), 1)
#         while True:
#             dataset = []
#             label = []
#             for i in range(BATCH_SIZE):
#                 point = np.random.randn(2) * std
#                 center = random.choice(centers)
#                 point[0] += center[0]
#                 point[1] += center[1]
#                 dataset.append(point)
#             dataset = np.array(dataset, dtype='float32')
#             dataset /= 1.414
#             yield dataset
#     if DATASET == "25gaussians":
#         dataset = []
#         for i in range(int(100000 / 25)):
#             for x in range(-2, 3):
#                 for y in range(-2, 3):
#                     point = np.random.randn(2) * std
#                     point[0] += 2*x
#                     point[1] += 2*y
#                     dataset.append(point)
#         dataset = np.array(dataset, dtype='float32')
#         np.random.shuffle(dataset)
#         dataset /= 2.828
#         while True:
#             for i in range(int(len(dataset) / BATCH_SIZE)):
#                 yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
#     if DATASET == "swissroll":
#         n = 20
#         a = 0.1
#         b = 0.1
#         delta_theta = 2*np.pi/12
#         centers_x = []
#         centers_y = []
#         for i in range(n):
#             centers_x.append((a + b * i * delta_theta) * np.cos(i * delta_theta))
#             centers_y.append((a + b * i * delta_theta) * np.sin(i * delta_theta))
#         centers_x = np.expand_dims(np.array(centers_x), 1)
#         centers_y = np.expand_dims(np.array(centers_y), 1)
#         centers = np.concatenate((centers_x, centers_y), 1)
#         while True:
#             dataset = []
#             for i in range(BATCH_SIZE):
#                 point = np.random.randn(2) * std
#                 center = random.choice(centers)
#                 point[0] += center[0]
#                 point[1] += center[1]
#                 dataset.append(point)
#             dataset = np.array(dataset, dtype='float32')
#             yield dataset

# # PLot real vs. fake samples of Mixture of Gaussians
# def plot_gaussian(z, x, netG, netQ, epoch=None, path=None):
#     x_hat = netG(z)  # fake samples
#     # z_hat = netQ(x)  # latent space
#     x = x.cpu().data.numpy()
#     # z = z.cpu().data.numpy()
#     x_hat = x_hat.cpu().data.numpy()
#     # z_hat = z_hat.cpu().data.numpy()
#     plt.figure(figsize=(9, 9))
#     plt.scatter(x_hat[:, 0], x_hat[:, 1])
#     plt.xlim(-2.25, 2.25)
#     plt.ylim(-2.25, 2.25)
#     if epoch is not None:
#         plt.title("Generated samples at epoch " + str(epoch), fontsize=25)
#     else:
#         plt.title("Generated samples")
#     if path is not None:
#         plt.savefig(path+"_"+str(epoch)+".png")
#     plt.close()
#     """
#     _, ax = plt.subplots(1, 2, figsize=(20, 9))
#     ax[0].scatter(x[:, 0], x[:, 1])
#     ax[0].set_title("Real Samples")
#     ax[1].scatter(x_hat[:, 0], x_hat[:, 1])
#     ax[1].set_title("Fake Samples")
#     if epoch is not None:
#         plt.title("At epoch " + str(epoch))
#         if path is not None:
#             plt.savefig(path+"_"+str(epoch)+".png")
#     plt.close()
#     """

# def plot_losses(losses_dict, keys=None, duality=True, path=None, step=10):
#     if keys is None:
#         keys = list(losses_dict.keys())
#     idxs = np.arange(step, len(losses_dict["primal loss"])+1, step)
#     idxs = np.append(1, idxs)
#     for key in keys:
#         plt.plot(idxs, np.array(losses_dict[key])[idxs-1], label=key)
#     if duality:
#         duality_loss = [primal - dual for (primal, dual) in zip(losses_dict["primal loss"], losses_dict["dual loss"])]
#         plt.plot(idxs, np.array(duality_loss)[idxs-1], label="Duality Gap")
#     plt.xlabel("epochs")
#     plt.ylabel("loss values")
#     plt.legend()
#     plt.savefig(path+".png")
#     plt.close()

# *************
# *** MNIST ***
# *************
# Data generator for MNIST dataset example
# def mnist_generator(data, batch_size, n_labelled, limit=None):
#     images, targets = data
#     rng_state = np.random.get_state()
#     np.random.shuffle(images)
#     np.random.set_state(rng_state)
#     np.random.shuffle(targets)
#     if limit is not None:
#         print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
#         images = images.astype('float32')[:limit]
#         targets = targets.astype('int32')[:limit]
#     if n_labelled is not None:
#         labelled = np.zeros(len(images), dtype='int32')
#         labelled[:n_labelled] = 1

#     def get_epoch():
#         rng_state = np.random.get_state()
#         np.random.shuffle(images)
#         np.random.set_state(rng_state)
#         np.random.shuffle(targets)
#         if n_labelled is not None:
#             np.random.set_state(rng_state)
#             np.random.shuffle(labelled)
#         image_batches = images.reshape(-1, batch_size, 784)
#         target_batches = targets.reshape(-1, batch_size)
#         if n_labelled is not None:
#             labelled_batches = labelled.reshape(-1, batch_size)
#             for i in range(len(image_batches)):
#                 yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))
#         else:
#             for i in range(len(image_batches)):
#                 yield (np.copy(image_batches[i]), np.copy(target_batches[i]))
#     return get_epoch

# def load(batch_size, test_batch_size, n_labelled=None):
#     with gzip.open('./datasets/MNIST/mnist.pkl.gz', 'rb') as f:
#         train_data, dev_data, test_data = pickle.load(f, encoding='latin1')
#     return (mnist_generator(train_data, batch_size, n_labelled),
#             mnist_generator(dev_data, test_batch_size, n_labelled),
#             mnist_generator(test_data, test_batch_size, n_labelled))

# def inf_train_gen_mnist(gen):
#     while True:
#         for images, targes in gen():
#             yield images

def filename(z_dim, structure_dim, batch_size, lambda_mmd, lambda_gp):
    return "_Z_" + str(z_dim) + \
        "_SD_" + str(structure_dim) + \
        "_BS_" + str(batch_size) + \
        "_LM_" + str(lambda_mmd).replace(".", "-") + \
        "_LG_" + str(lambda_gp).replace(".", "-") + ".png"

def picture(z, x, netG, netQ, FN, picture_type):
    if picture_type == "fake":
        fake = netG(z).cpu().data.numpy()
        plt.scatter(fake[:, 0], fake[:, 1])
        plt.savefig("fake"+FN)
        plt.close()
    if picture_type == "latent":
        latent = netQ(x).cpu().data.numpy()
        plt.scatter(latent[:, 0], latent[:, 1])
        plt.savefig("latent"+FN)
        plt.close()
    if picture_type == "post":
        post = netG(netQ(x)).cpu().data.numpy()
        real = x.cpu().data.numpy()
        plt.scatter(real[:, 0], real[:, 1])
        plt.scatter(post[:, 0], post[:, 1])
        plt.savefig("post"+FN)
        plt.close()

def all_pictures(z, x, netG, netQ, DIR, FN, MMD, GP, RE):
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    fake = netG(z).cpu().data.numpy()
    plt.scatter(fake[:, 0], fake[:, 1])
    plt.title("Generated Samples")
    plt.subplot(2, 2, 2)
    latent = netQ(x).cpu().data.numpy()
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.title("Latent Space")
    plt.subplot(2, 2, 3)
    post = netG(netQ(x)).cpu().data.numpy()
    real = x.cpu().data.numpy()
    plt.scatter(real[:, 0], real[:, 1], label="Real Samples")
    plt.scatter(post[:, 0], post[:, 1], label="Post Samples")
    plt.legend()
    plt.title("Post Samples vs. Real Samples")
    plt.subplot(2, 2, 4)
    plt.plot(MMD, label="MMD")
    plt.plot(GP, label="GP")
    plt.plot(RE, label="RE")
    plt.legend()
    plt.savefig(DIR + "all" + FN)
    plt.close()

def save_models(net_dict, epoch, path):
    for key in net_dict.keys():
        torch.save(net_dict[key].state_dict(), path+str(key)+"_"+str(epoch))

def train_al(netI, netG, netD, optim_I, optim_G, optim_D,
             train_gen, train_loader, batch_size, start_epoch, end_epoch, 
             z_dim, device, lab, present_label, all_label, 
             lambda_gp, lambda_power, sample_sizes = None, 
             critic_iter = 10, critic_iter_d = 10, lambda_mmd = 10.0):

    if sample_sizes is None:
        sample_sizes = [int(len(train_loader.dataset.indices) / len(present_label))] * (len(present_label) - 1)

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
        iter_count = 0
        for _ in range(critic_iter):
            images, _ = next_batch(data, train_loader)
            x = images.view(len(images), 784).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            fake_x = netG(z)
            netI.zero_grad()
            netG.zero_grad()
            cost_GI = GI_loss(netI, netG, netD, z, fake_z)
            images, _ = next_batch(data, train_loader)
            x = images.view(len(images), 784).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            ## MMD loss has been removed from the paper
            fake_z = netI(x)
            mmd = mmd_penalty(fake_z, z, kernel="IMQ")
            primal_cost = cost_GI + lambda_mmd * mmd
            primal_cost.backward()
            optim_I.step()
            optim_G.step()
        # print('GI: '+str(primal(netI, netG, netD, real_data).cpu().item()))
        # print('GI: '+str(cost_GI.cpu().item()))
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
            x = images.view(len(images), 784).to(device)
            z = torch.randn(len(images), z_dim).to(device)
            fake_z = netI(x)
            fake_x = netG(z)
            netD.zero_grad()
            cost_D = D_loss(netI, netG, netD, z, fake_z)
            images, y = next_batch(data, train_loader)
            x = images.view(len(images), 784)
            x = x.to(device)
            z = torch.randn(len(images), z_dim)
            z = z.to(device)
            fake_z = netI(x)
            gp_D = gradient_penalty_dual(x.data, z.data, netD, netG, netI)
            dual_cost = cost_D + lambda_gp * gp_D
            dual_cost.backward()
            optim_D.step()
            # loss_mmd.append(mmd.cpu().item())
        # print('D: '+str(primal(netI, netG, netD, real_data).cpu().item()))
        # print('D: '+str(cost_D.cpu().item()))
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
                if torch.is_tensor(train_gen.targets):
                    temp = torch.where(train_gen.targets == cur_lab)[0] 
                else:
                    temp = torch.where(torch.Tensor(train_gen.targets) == cur_lab)[0] 
                idxs2 = torch.cat([idxs2, temp[np.random.choice(len(temp), sample_sizes[count], replace=False)]])
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
            
            z = torch.ones(bs, z_dim, 1, 1, device = device) * 3
            x = x.to(device) 
            fake_z = netI(x)

            netI.zero_grad()
            loss_power = lambda_power * I_loss(fake_z, z.reshape(bs, z_dim))
            loss_power.backward()

            optim_I.step()




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
#     plt.savefig('size_power.pdf', dpi=150)
    plt.show()

def next_batch(data_iter, train_loader):
    try:
        return next(data_iter)
    except StopIteration:
        # Reset the iterator and return the next batch
        return next(iter(train_loader))