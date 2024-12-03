import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class NoisyMNIST(Dataset):
    def __init__(self, noise_level: float = 0.1, train: bool = True):
        dataset = MNIST(root="data", download=True, transform=ToTensor(), train=train)
    
        self.dataset = dataset
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x_noisy = x + self.noise_level * torch.randn_like(x)
        return x_noisy.float(), x.float()
    
    def __repr__(self):
        return f"NoisyMNIST(noise_level={self.noise_level})"
