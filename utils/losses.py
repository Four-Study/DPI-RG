import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd as autograd
from torch.autograd import grad
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

gpu = 0

# loss function for D update:
def D_loss(netI, netG, netD, z, fake_z):
    post_z = netI(netG(z))
    losses = netD(fake_z) - netD(post_z)
    return losses.mean()

# loss function for G and I update:
def GI_loss(netI, netG, netD, z, fake_z, p=2):
    post_z = netI(netG(z))
    n_dim = len(post_z.shape)
    dim = list(range(1, n_dim))
    # distance = torch.dist(real_data, post_data, p=p)
    # sz = 1
    l2 = torch.sqrt(torch.sum((z-post_z)**2, dim=dim))
    # sz = real_data.shape[0]
    losses = l2 + netD(post_z) - netD(fake_z)
    return losses.mean()

# reconstruction for z
def rec_z(netG, netI, z):
    z_t = netI(netG(z))
    n_dim = len(z_t.shape)
    dim = list(range(1, n_dim))
    l2 = torch.sqrt(torch.sum((z-z_t)**2, dim=dim))
    return l2.mean()

# primal loss function
def primal(netI, netG, netD, z, p=2):
    post_z = netI(netG(z))
    n_dim = len(post_z.shape)
    dim = list(range(1, n_dim))
    l2 = torch.sqrt(torch.sum((z-post_z)**2, dim=dim))
    return l2.mean()

# primal loss based on z_sample
# def primal_z(netG, real_data, z_sample, p=2):
#     fake_data = netG(z_sample)
#     n_dim = len(post_data.shape)
#     dim = list(range(1, n_dim))
#     l2 = torch.sqrt(torch.sum((real_data-fake_data)**2, dim=dim))
#     return l2.mean()


# dual loss function
def dual(netI, netG, netD, z, fake_z):
    losses = netD(z) - netD(fake_z)
    return losses.mean()

# *** Penalty Terms ***
# Gradient Penalty
# gradient penalty function
def _gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda(gpu)
    z = x + alpha * (y - x)
    # gradient penalty
    z = Variable(z, requires_grad=True).cuda(gpu)
    o = f(z)
    g = grad(o, z, grad_outputs=(torch.ones(o.size())).cuda(gpu), create_graph=True)[0].view(z.size(0), -1)
    # g = grad(o, z, grad_outputs=(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp

# gradient penalty for netD
def gradient_penalty_D(x, z, netD, netG, netI):
    x_hat = netG(z)
    x_tilde = netG(netI(x))
    return _gradient_penalty(x_hat, x_tilde, netD)

# gradient penalty for f(G(.))

def gradient_penalty_DG(x, z, netD, netG, netI):
    def _g(x):
        return netD(netG(x))
    z_hat = netI(x)
    return _gradient_penalty(z_hat, z, _g)

# gradient penalty for dual
def gradient_penalty_dual(x, z, netD, netG, netI):
    # x_hat = netG(z)
    z_hat = netI(x)
    return _gradient_penalty(z_hat, z, netD)

# Penalty for z and I(x)
def z_Qx(x, z, netD, netG, netI):
    return ((z - netI(x)).norm(p=2, dim=1)**2).mean()

# *** MMD penalty ***
# MMD loss between z and I(x)
def mmd_penalty(z_hat, z, kernel="RBF", sigma2_p=1):
    n = z.shape[0]
    zdim = z.shape[1]
    half_size = int((n * n - n)/2)
    #
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    #
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    #
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    #
    if kernel == "RBF":
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
        #
        res1 = torch.exp(-dists_zh/2./sigma2_k)
        res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
        res1 = torch.mul(res1, 1. - torch.eye(n).cuda())
        res1 = res1.sum() / (n*n-n)
        res2 = torch.exp(-dists/2./sigma2_k)
        res2 = res2.sum()*2./(n*n)
        stat = res1 - res2
        return stat
    #
    elif kernel == "IMQ":
        Cbase = 2 * zdim * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, 1. - torch.eye(n).cuda())
            res1 = res1.sum() / (n*n-n)
            res2 = C / (C + dists)
            res2 = res2.sum()*2./(n*n)
            stat = stat + res1 - res2
        return stat

# *** Power Penalty ***
# power penalty for netD
def power_penalty_D(x, eta, netI):
    z_hat = netG(x)
    target = eta * torch.ones(len(z_hat))
    l2 = torch.norm(z_hat - target)
    return l2