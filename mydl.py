import torch
import torch.nn as nn
import torch.nn.functional as F

# models for four basic image datasets, Mnist and FashionMnist are
# gray scale image, while Cifar10 and Cifar100 are RGB images.
# models are all simple enough that you can even implement these
# using only torch.nn.sequential()

# https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_1_cnn_convnet_mnist/
class Conv2dMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 28x28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x: torch.Tensor):
        in_size = x.size(0)
        x = self.conv1(x) # 24
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2, 2) # 12
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1) # perform softmax on sample dim
        
        return x
    
# https://zhuanlan.zhihu.com/p/391444296
class Conv2dFashionMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), 2),
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
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = F.max_pool2d(x, 2, 2) # 14 * 14
        x = self.layer2(x) # 12 * 12
        x = self.layer3(x) # 10 * 10
        x = F.max_pool2d(x, 2, 2) # 5 * 5
        x = x.view(x.size(0), -1) # [b, f]
        x = self.fc(x) # [b, 10]
        return x # logits

# https://blog.csdn.net/shi2xian2wei2/article/details/84308644
class Conv2dCIFAR10(nn.Module):
    '''
    RGB image is more difficult than gray scale, so use fully-
    connected layer as little as possible as they are not that
    efficient when passing features extracted from layers.
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x: torch.Tensor):
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn1(F.leaky_relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.leaky_relu(self.conv3(x)))
        x = self.bn2(F.leaky_relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.leaky_relu(self.conv5(x)))
        x = self.bn3(F.leaky_relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def name2class():
    import inspect
    import importlib

    current_file_path = inspect.getfile(inspect.currentframe())
    module_name = inspect.getmodulename(current_file_path)
    module = importlib.import_module(module_name)

    model_map = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            model_map[name] = obj

    return model_map
MODEL = name2class()
