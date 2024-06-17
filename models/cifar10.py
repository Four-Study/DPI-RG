import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# class I_CIFAR10(models.ResNet):
#     def __init__(self, nz=5, img_size=32):
#         # Initialize with the basic block and layer configuration of ResNet-18
#         super(I_CIFAR10, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=nz)
#         # Initialize with the basic block and layer configuration of ResNet-50
#         # super(I_CIFAR10, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=nz)
#         self.img_size = img_size
#     def forward(self, x):
#         # Resize the input
#         x = x.view(-1, 3, self.img_size, self.img_size)
#         # Call the original forward method
#         return super(I_CIFAR10, self).forward(x)
        

class I_CIFAR10(models.ResNet):
    def __init__(self, nz=5, img_size=32):
        # Initialize with the basic block and layer configuration of ResNet-18
        super(I_CIFAR10, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=nz)
        self.conv1 = nn.Conv2d(3, self.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_size = img_size
    def forward(self, x):
        x = x.view(-1, 3, self.img_size, self.img_size)
        return super(I_CIFAR10, self).forward(x)

        
# class I_CIFAR10_2(models.ResNet):
#     def __init__(self, nz=5, img_size=64):
#         # Initialize with the basic block and layer configuration of ResNet-34
#         super(I_CIFAR10_2, self).__init__(block=models.resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=nz)
#         # Initialize with the basic block and layer configuration of ResNet-50
#         # super(I_CIFAR10, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=nz)
#         self.img_size = img_size
#     def forward(self, x):
#         # Resize the input
#         x = x.view(-1, 3, self.img_size, self.img_size)
#         # Call the original forward method
#         return super(I_CIFAR10_2, self).forward(x)


class I_CIFAR10_2(models.ResNet):
    def __init__(self, nz=5, img_size=64):
        # Initialize with the basic block and layer configuration of ResNet-34
        super(I_CIFAR10_2, self).__init__(block=models.resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=nz)
        self.conv1 = nn.Conv2d(3, self.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.img_size = img_size
    def forward(self, x):
        x = x.view(-1, 3, self.img_size, self.img_size)
        return super(I_CIFAR10_2, self).forward(x)


class I_CIFAR10_3(nn.Module):
    def __init__(self, nz=5, img_size=32):
        super(I_CIFAR10_3, self).__init__()
        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16()
        
        # Replace the classifier part of VGG16
        # VGG16 originally outputs 1000 classes for ImageNet
        # We change it to output nz classes
        self.vgg16.classifier[6] = nn.Linear(4096, nz)
        self.img_size = img_size
    def forward(self, x):
        # Resize the input if needed
        x = x.view(-1, 3, self.img_size, self.img_size)
        # Call the forward method of VGG16
        return self.vgg16(x)
    
class G_CIFAR10(nn.Module):
    def __init__(self, nz, nc=3, ngf=16, img_size=32):
        super(G_CIFAR10, self).__init__()
        self.img_size = img_size
        self.nz = nz
        layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) 
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]
        if img_size == 32:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ])
        elif img_size == 64:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        output = self.main(input)
        return output.view(-1, 3, self.img_size**2)

# class D_MNIST(nn.Module):
#     def __init__(self, ngpu=1, nc=1, ndf=32):
#         super(D_MNIST, self).__init__()
#         self.ngpu = ngpu
#         layers = [
#             # input is (nc) 
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) 
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) 
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True)
#         ]
#         if nc == 3:
#             layers.extend([
#                 # state size. (ndf*8)
#                 nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#                 nn.BatchNorm2d(ndf * 8),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # state size. (ndf*8) 
#                 nn.Conv2d(ndf * 8,       1, 4, 2, 1, bias=False),
#                 nn.Sigmoid()
#             ])
#         else:
#             layers.extend([
#                 nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
#                 nn.Sigmoid()
#             ])
#         self.main = nn.Sequential(*layers)

#     def forward(self, input):
#         input = input.view(-1, 1, 28, 28)
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         return output.view(-1, 1).squeeze(1)
    
class D_CIFAR10(nn.Module):
    def __init__(self, nz, ndf = 16, power = 6):
        super(D_CIFAR10, self).__init__()
        self.power = power
        layers = [
            # input is (nz) 
            # state size. (ndf * 4) 
            nn.Linear(nz * power , ndf * 4),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf * 2) 
            nn.Linear(ndf * 4, ndf * 2),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            # state size. (ndf) 
            nn.Linear(ndf * 2, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        # print(input.shape)
        # dist = nn.PairwiseDistance(input, p=2)
        powers = [i for i in range(0, self.power)]
        input = torch.cat([torch.pow(input, i) for i in powers], dim=1)
        # print(powers.shape)
        output = self.main(input)
        return output.squeeze(1)