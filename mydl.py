import torch
import torch.nn as nn
import torch.nn.functional as F

# https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/
class MyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 28x28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x: torch.Tensor):
        in_size = x.size(0)
        out = self.conv1(x) # 24
        out = F.relu(out)
        
        out = F.max_pool2d(out, 2, 2) # 12
        
        out = self.conv2(out)
        out = F.relu(out)
        
        out = out.view(in_size, -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # out = F.softmax(out, dim=1) # perform softmax on sample dim
        
        return out
