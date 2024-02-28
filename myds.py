import torch
import torch.nn.functional as th_F
import torch.utils.data.dataset as th_dataset

from typing import Any

class MyDataset(th_dataset.Dataset):
    def __init__(self, data: torch.Tensor, gdth: torch.Tensor) -> None:
        super().__init__()
        self.data = data
        self.gdth = gdth
        
        self._preprocess()
    
    def __getitem__(self, index) -> Any:
        return self.data[index], self.gdth[index]

    def __len__(self):
        return self.data.size(0)

    def _preprocess(self):
        self.data = self.data.float()
        min_val = self.data.min()
        max_val = self.data.max()
        self.data = (self.data - min_val) / (max_val - min_val)
        
        self.gdth = th_F.one_hot(self.gdth, 10)
