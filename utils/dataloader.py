import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# MNIST dataloader
class MNISTBatcher(Dataset):
    def __init__(self, data_path='/modesim/scratch/data/MNIST/mnist.pkl.gz', train=True):
        self.data_path = data_path
        self.train = train
        with gzip.open(self.data_path, 'rb') as f:
            train_data, dev_data, test_data = pickle.load(f, encoding='latin1')
        if self.train:
            self.data = train_data
        else:
            self.data = test_data

    def __len__(self):
        return len(self.data[1])

    def __getitem__():
        images, targets = self.data
        return {"images": images, "targets": targets}

# CelebA dataloader
class CelebABatcher(Dataset):
    def __init__(self, data_path, resolution=64, transform=None):
        self.data_path = data_path
        self.resolution = resolution
        self.image_files = [entry for entry in os.listdir(self.data_path+str(self.resolution))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.data_path+str(self.resolution)+"/"+self.image_files[idx]))
        # image = image.reshape((-1, )+image.shape)
        if self.transform:
            image = self.transform(image)
        return image
