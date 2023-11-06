import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
import gzip
import pickle

import torch

random.seed(1)
np.random.seed(1)
matplotlib.use('Agg')

# ****************************
# *** Mixture of Gaussians ***
# ****************************
# Data simulator for mixture of gaussians example
def inf_train_gen(DATASET, BATCH_SIZE=1024, std=0.02):
    if DATASET == "8gaussians":
        n = 8
        radius = 2.
        delta_theta = 2*np.pi/n
        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append(radius * np.cos(i * delta_theta))
            centers_y.append(radius * np.sin(i * delta_theta))
        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)
        centers = np.concatenate((centers_x, centers_y), 1)
        while True:
            dataset = []
            label = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * std
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414
            yield dataset
    if DATASET == "25gaussians":
        dataset = []
        for i in range(int(100000 / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * std
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828
        while True:
            for i in range(int(len(dataset) / BATCH_SIZE)):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    if DATASET == "swissroll":
        n = 20
        a = 0.1
        b = 0.1
        delta_theta = 2*np.pi/12
        centers_x = []
        centers_y = []
        for i in range(n):
            centers_x.append((a + b * i * delta_theta) * np.cos(i * delta_theta))
            centers_y.append((a + b * i * delta_theta) * np.sin(i * delta_theta))
        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)
        centers = np.concatenate((centers_x, centers_y), 1)
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * std
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            yield dataset

# PLot real vs. fake samples of Mixture of Gaussians
def plot_gaussian(z, x, netG, netQ, epoch=None, path=None):
    x_hat = netG(z)  # fake samples
    # z_hat = netQ(x)  # latent space
    x = x.cpu().data.numpy()
    # z = z.cpu().data.numpy()
    x_hat = x_hat.cpu().data.numpy()
    # z_hat = z_hat.cpu().data.numpy()
    plt.figure(figsize=(9, 9))
    plt.scatter(x_hat[:, 0], x_hat[:, 1])
    plt.xlim(-2.25, 2.25)
    plt.ylim(-2.25, 2.25)
    if epoch is not None:
        plt.title("Generated samples at epoch " + str(epoch), fontsize=25)
    else:
        plt.title("Generated samples")
    if path is not None:
        plt.savefig(path+"_"+str(epoch)+".png")
    plt.close()
    """
    _, ax = plt.subplots(1, 2, figsize=(20, 9))
    ax[0].scatter(x[:, 0], x[:, 1])
    ax[0].set_title("Real Samples")
    ax[1].scatter(x_hat[:, 0], x_hat[:, 1])
    ax[1].set_title("Fake Samples")
    if epoch is not None:
        plt.title("At epoch " + str(epoch))
        if path is not None:
            plt.savefig(path+"_"+str(epoch)+".png")
    plt.close()
    """

def plot_losses(losses_dict, keys=None, duality=True, path=None, step=10):
    if keys is None:
        keys = list(losses_dict.keys())
    idxs = np.arange(step, len(losses_dict["primal loss"])+1, step)
    idxs = np.append(1, idxs)
    for key in keys:
        plt.plot(idxs, np.array(losses_dict[key])[idxs-1], label=key)
    if duality:
        duality_loss = [primal - dual for (primal, dual) in zip(losses_dict["primal loss"], losses_dict["dual loss"])]
        plt.plot(idxs, np.array(duality_loss)[idxs-1], label="Duality Gap")
    plt.xlabel("epochs")
    plt.ylabel("loss values")
    plt.legend()
    plt.savefig(path+".png")
    plt.close()

# *************
# *** MNIST ***
# *************
# Data generator for MNIST dataset example
def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)
        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)
        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)
        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)
            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))
        else:
            for i in range(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))
    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    with gzip.open('./datasets/MNIST/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f, encoding='latin1')
    return (mnist_generator(train_data, batch_size, n_labelled),
            mnist_generator(dev_data, test_batch_size, n_labelled),
            mnist_generator(test_data, test_batch_size, n_labelled))

def inf_train_gen_mnist(gen):
    while True:
        for images, targes in gen():
            yield images

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
