import torch
import torch.nn as nn
import torch.nn.functional as F

# https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/
class Conv2dMnist(nn.Module):
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
    
# https://zhuanlan.zhihu.com/p/391444296
class Conv2dFashionMnist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # 28 * 28 -> max_pool2d(2, 2) -> 14 * 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # 12 * 12 -> max_poold(2, 2) -> 6 * 6
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x: torch.Tensor):
        out = x
        out = self.layer1(out)
        out = F.max_pool2d(out, 2, 2) # 14 * 14
        out = self.layer2(out) # 12 * 12
        out = self.layer3(out) # 10 * 10
        out = F.max_pool2d(out, 2, 2) # 5 * 5
        out = out.view(out.size(0), -1) # [b, f]
        out = self.fc(out) # [b, 10]
        return out # logits
