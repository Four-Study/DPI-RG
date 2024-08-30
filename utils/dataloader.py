import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(dataset_name, root="./datasets", train=True, transform=None, download=True):
    """
    Get a dataset from torchvision.datasets.
    
    Args:
    dataset_name (str): Name of the dataset (e.g., 'FashionMNIST', 'CIFAR10', etc.)
    root (str): Root directory of dataset where data will be stored
    train (bool): If True, creates dataset from training set, otherwise creates from test set
    transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version
    download (bool): If true, downloads the dataset from the internet and puts it in root directory
    
    Returns:
    dataset: A torchvision dataset
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(root=root, train=train, transform=transform, download=download)
    
    return dataset

def get_data_loader(dataset, labels, batch_size, shuffle=True):
    """
    Get a DataLoader for the specified labels from the dataset.
    
    Args:
    dataset: A torchvision dataset
    labels (list): List of labels to include in the DataLoader
    batch_size (int): How many samples per batch to load
    shuffle (bool): If True, shuffle the data
    
    Returns:
    DataLoader: A DataLoader containing only the specified labels
    """
    if torch.is_tensor(dataset.targets):
        idxs = torch.where(torch.isin(dataset.targets, torch.tensor(labels)))[0]
    else:
        idxs = torch.where(torch.isin(torch.tensor(dataset.targets), torch.tensor(labels)))[0]
    
    subset = Subset(dataset, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)