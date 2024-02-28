import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from attribute_mapping import AttributeMapping

import torch

import torch.nn as th_nn
import torch.optim as th_optim
import torch.nn.functional as th_F

import torch.utils.data
import torch.utils.data.dataloader as th_dataloader
import torch.utils.data.dataset as th_dataset
import torchvision

import myds
import mydl

import tensorboardX as tfx

def get_mnist_dataset():
    download = False
    if not osp.exists("data"):
        download = True
    
    return torchvision.datasets.MNIST(root="data", download=download)

class Trainer:
    def __init__(self, dataset: th_dataset.Dataset, config: map) -> None:
        # config map should contain:
        #   epoch
        #   batch_size
        #   train/eval/test ratio
        self.config = AttributeMapping(config)
        
        self.model = None
        self.rt_device = None
        self.optimizer = None
        self.scheduler = None
        
        split_non = myds.MyDataset(data, gdth)
        split_train, split_eval, split_test = torch.utils.data.random_split(
            split_non, [
                int(len(split_non)*0.6),
                int(len(split_non)*0.1),
                int(len(split_non)*0.3)
            ]
        )
        self.dataloader_train = th_dataloader.DataLoader(
            split_train,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.dataloader_eval = th_dataloader.DataLoader(
            split_eval,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.dataloader_test = th_dataloader.DataLoader(
            split_test,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.log_writer = tfx.SummaryWriter("log")
    
    def train(self):
        self.rt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = th_optim.Adam(self.model.parameters())
        self.scheduler = th_optim.lr_scheduler.ExponentialLR(gamma=0.9)
        self.model = mydl.MyConvNet().to(self.rt_device)
        
        for epoch in range(self.config.epoch):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            
            self.scheduler.step()
    
    def train_epoch(self, epoch: int):
        self.model.train()
        for batch, (data, gdth) in tqdm(enumerate(self.dataloader_train), desc=f"epoch {epoch}", ncols=100):
            data, gdth = data.to(self.rt_device), gdth.to(self.rt_device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = th_F.cross_entropy(pred, gdth)
            loss.backward()
            
            loss = loss.item()
            accu = pred.eq(gdth).int().sum().item()
            if (batch+1)%self.config.print_gap == 0:
                tqdm.write(f"[train] loss: {loss}, accu: {accu}")
                self.log_writer.add_scalar("train/loss", scalar_value=loss, global_step=epoch*len(self.dataloader_train)+batch)
                self.log_writer.add_scalar("train/accu", scalar_value=accu, global_step=epoch*len(self.dataloader_train)+batch)
        
    def test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_accu = 0.0
        with torch.no_grad():
            for data, gdth in self.dataloader_test:
                data, gdth = data.to(self.rt_device), gdth.to(self.rt_device)
                pred = self.model(data)
                total_loss += th_F.cross_entropy(pred, gdth).item()
                total_accu += pred.eq(gdth).int().sum().item()
        loss = total_loss / len(self.dataloader_test)
        accu = total_accu / len(self.dataloader_test)
        print(f"[test] loss: {loss}, accu: {accu}")
        self.log_writer.add_scalar(tag="test/loss", scalar_value=loss, global_step=(epoch+1)*len(self.dataloader_train))
        self.log_writer.add_scalar(tag="test/accu", scalar_value=accu, global_step=(epoch+1)*len(self.dataloader_train))

def train(model: th_nn.Module, dataset: th_dataset.Dataset, cfg: map):
    cfg = AttributeMapping(cfg) # convert dict to attr
    
    rt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = th_optim.Adam(model.parameters())
    scheduler = th_optim.lr_scheduler.ExponentialLR(gamma=0.9)
    dataloadr = th_dataloader.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    
    model.to(rt_device)
    model.train()
    for epoch_idx in range(cfg.epoch):
        for batch_idx, (data, gdth) in tqdm(enumerate(dataloadr), desc=f"epoch {epoch_idx+1}", ncols=100):
            data, gdth = data.to(rt_device), gdth.to(rt_device)
            
            optimizer.zero_grad()
            
            pred = model(data)
            loss = th_F.cross_entropy(pred, gdth)
            loss.backward()
            
            if (epoch_idx + 1) % cfg.print_interval == 0:
                tqdm.write(f"{batch_idx+1}/{len(dataloadr)} loss: {loss.item()}")
                log_writer.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=epoch_idx*len(dataloadr)+batch_idx)
        scheduler.step()

def test(model: th_nn.Module, dataset: th_dataset.Dataset, cfg: map):
    rt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloadr = th_dataloader.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    
    total_loss = 0.0
    total_accu = 0
    model.to(rt_device)
    model.eval()
    with torch.no_grad():
        for data, gdth in dataloadr:
            data, gdth = data.to(rt_device), gdth.to(rt_device)
            pred = model(data)
            loss = th_F.cross_entropy(pred, gdth)
            accu = pred.eq(gdth).int().sum().item()
            
            total_loss += loss
            total_accu += accu
    
    avg_loss = total_loss.item() / len(dataloadr)
    avg_accu = total_accu.item() / len(dataloadr)
    print(f"[test] loss: {avg_loss}, accu: {avg_accu}")
    log_writer.add_scalar("test/loss", scalar_value=avg_loss, global_step=cfg.global_step)
    log_writer.add_scalar("test/accu", scalar_value=avg_accu, global_step=cfg.global_step)


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    
    mnist_raw = get_mnist_dataset()
    data = mnist_raw.data
    gdth = mnist_raw.targets
    
    
    
    
