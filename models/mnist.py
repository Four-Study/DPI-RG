import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
DIM_MNIST = 128
Z_DIM_MNIST = 8
Z_OUTPUT_DIM_MNIST = 784

class I_MNIST(nn.Module):
    def __init__(self):
        super(I_MNIST, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM_MNIST, 2*DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM_MNIST, 4*DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM_MNIST, Z_DIM_MNIST)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM_MNIST)
        out = self.output(out)
        return out

class G_MNIST(nn.Module):
    def __init__(self):
        super(G_MNIST, self).__init__()
        # Define Layers
        preprocess = nn.Sequential(
            nn.Linear(Z_DIM_MNIST, 4*4*4*DIM_MNIST),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM_MNIST, 2*DIM_MNIST, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM_MNIST, DIM_MNIST, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM_MNIST, 1, 8, stride=2)
        # Define Network Layers
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    # Define forward function
    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM_MNIST, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, Z_OUTPUT_DIM_MNIST)

class D_MNIST(nn.Module):
    def __init__(self):
        super(D_MNIST, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM_MNIST, 2*DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM_MNIST, 4*DIM_MNIST, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM_MNIST, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM_MNIST)
        out = self.output(out)
        return out.view(-1)

class Class_MNIST(nn.Module):
    def __init__(self):
        super(Class_MNIST, self).__init__()
        main = nn.Sequential(
            nn.Linear(Z_DIM_MNIST, 200),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(200, 10)

    def forward(self, input):
        out = self.main(input)
        out = self.output(out)
        out = F.log_softmax(out, dim=1)
        return out
