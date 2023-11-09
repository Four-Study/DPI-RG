import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
    
class I_MNIST(nn.Module):

    def __init__(self, nz, ngpu=1, nc=1):
        super(I_MNIST, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        if nc == 3:
            ks = 5
        else:
            ks = 4
        self.main = nn.Sequential(            
            nn.Conv2d(in_channels=nc, out_channels=6, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=ks, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=nz)
        )

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    
class G_MNIST(nn.Module):
    def __init__(self, nz, ngpu=1, nc=1, ngf=32):
        super(G_MNIST, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
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
        if nc == 3:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
                nn.Tanh()
            ])
        else:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=False),
                nn.Tanh()
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 28*28)

class D_MNIST(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf=32):
        super(D_MNIST, self).__init__()
        self.ngpu = ngpu
        layers = [
            # input is (nc) 
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) 
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if nc == 3:
            layers.extend([
                # state size. (ndf*8)
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) 
                nn.Conv2d(ndf * 8,       1, 4, 2, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            layers.extend([
                nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
                nn.Sigmoid()
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
class DiscriminatorB(nn.Module):
    def __init__(self, ngpu, nz, ndf = 128):
        super(DiscriminatorB, self).__init__()
        self.ngpu = ngpu
        layers = [
            # input is (nz) 
            # state size. (ndf * 4) 
            nn.Linear(nz, ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            # state size. (ndf * 2) 
            nn.Linear(ndf * 4, ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            # state size. (ndf) 
            nn.Linear(ndf * 2, ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.squeeze(1)
    
class VGG16(nn.Module):

    def __init__(self, ngpu, nc, nz):
        super(VGG16, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.ngpu = ngpu

        layers = [
            # block 1
            nn.Conv2d(in_channels=nc,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            # block 2
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            # block 3
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            # block 4
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            # block 5
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            # classifier
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, nz),
        ]
        self.main = nn.Sequential(*layers)

#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.detach().zero_()

    def forward(self, x):

        if x.is_cuda and self.ngpu > 1:
            logits = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            logits = self.main(x)
        return logits
