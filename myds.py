import torch
import torch.nn.functional as th_F
import torch.utils.data.dataset as th_dataset
import torchvision

from typing import Any

class Inner(th_dataset.Dataset):
    # MNIST
    # FashionMNIST
    # CIFAR10
    # CIFAR100
    def __init__(self, name: str, num_cls: int=-1, root: str="data", train: bool=True, download: bool=False) -> None:
        super().__init__()
        dataset_type = getattr(torchvision.datasets, name)
        rawd = dataset_type(root=root, train=train, download=download)
        self.data = rawd.data
        self.gdth = rawd.targets
        
        self._preprocess(num_cls)
        
    def __getitem__(self, index) -> Any:
        return self.data[index], self.gdth[index]

    def __len__(self):
        return self.data.size(0)
    
    def _preprocess(self, num_cls: int):
        self.data = self.data.float()
        min_val = self.data.min()
        max_val = self.data.max()
        self.data = (self.data - min_val) / (max_val - min_val)
        
        self.gdth = th_F.one_hot(self.gdth, num_cls)

# class Mnist(th_dataset.Dataset):
#     def __init__(self, root: str="data", download: bool=False) -> None:
#         super().__init__()
#         mnist = torchvision.datasets.MNIST(root=root, download=download)
#         self.data = mnist.data
#         self.gdth = mnist.targets
        
#         self._preprocess()
    
#     def __getitem__(self, index) -> Any:
#         return self.data[index], self.gdth[index]

#     def __len__(self):
#         return self.data.size(0)

#     def _preprocess(self):
#         self.data = self.data.float()
#         min_val = self.data.min()
#         max_val = self.data.max()
#         self.data = (self.data - min_val) / (max_val - min_val)
        
#         self.gdth = th_F.one_hot(self.gdth, 10)

# class FashionMnist(th_dataset.Dataset):
#     def __init__(self, root: str, download: bool=False) -> None:
#         super().__init__()
#         fashion_mnist = torchvision.datasets.FashionMNIST(root, download=download)
#         self.data = fashion_mnist.data
#         self.gdth = fashion_mnist.targets
        
#         self._preprocess()
        
#     def __getitem__(self, index) -> Any:
#         return self.data[index], self.gdth[index]
    
#     def __len__(self):
#         return self.data.size(0)
    
#     def _preprocess(self):
#         self.data = self.data.float()
#         min_val = self.data.min()
#         max_val = self.data.max()
#         self.data = (self.data - min_val) / (max_val - min_val)
        
#         self.gdth = th_F.one_hot(self.gdth, 10)
