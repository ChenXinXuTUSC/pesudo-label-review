import torch
import torch.nn.functional as th_F
import torch.utils.data.dataset as th_dataset
import torchvision

from typing import Any

class Inner(th_dataset.Dataset):
    # MNIST 32x32
    # FashionMNIST 64x64
    # CIFAR10
    # CIFAR100
    def __init__(
            self,
            name: str,
            num_cls: int=-1,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda x: x if type(x) is torch.Tensor else torch.tensor(x)),
                torchvision.transforms.Lambda(lambda x: x if len(x.shape) == 3 else x.unsqueeze(0)), # in case gray-scale image
                torchvision.transforms.Lambda(lambda x: x if x.size(0) != x.size(1) else x.permute(2, 0, 1)), # HWC -> CHW
                torchvision.transforms.Normalize(0.0, 1.0)
            ]),
            root: str="data",
            train: bool=True,
            download: bool=False
        ) -> None:
        super().__init__()
        dataset_type = getattr(torchvision.datasets, name)
        rawd = dataset_type(root=root, train=train, download=download)
        self.data = rawd.data
        self.gdth = rawd.targets

        # some other preparation
        if type(self.data) is not torch.Tensor:
            self.data = torch.tensor(self.data) # do not do this if the original type is torch.Tensor already
        if type(self.gdth) is not torch.Tensor:
            self.gdth = torch.tensor(self.gdth)
        if len(self.gdth.shape) == 1:
            # convert to one hot
            self.gdth = th_F.one_hot(self.gdth, num_cls)
        
        self.transform = transform
        
    def __getitem__(self, index) -> Any:
        x = self.data[index].float()
        if self.transform is not None:
            x = self.transform(x)
        y = self.gdth[index].float()
        return x, y

    def __len__(self):
        return len(self.gdth)
