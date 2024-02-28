import os
import os.path as osp
from datetime import datetime
from tqdm import tqdm
from attribute_mapping import AttributeMapping

import torch
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

        split_train, split_eval, split_test = torch.utils.data.random_split(
            dataset, [
                int(len(dataset)*self.config.ratio_train),
                int(len(dataset)*self.config.ratio_eval),
                int(len(dataset)*self.config.ratio_test)
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
        
        self.log_writer = tfx.SummaryWriter(self.config.logd_dir)
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def train(self):
        self.rt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mydl.MyConvNet().to(self.rt_device)
        self.optimizer = th_optim.Adam(self.model.parameters(), self.config.lr)
        self.scheduler = th_optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)
        
        loss_best = float(1 << 31)
        for epoch in range(self.config.epoch):
            self.train_epoch(epoch)
            self.scheduler.step()
            
            loss, _ = self.test_epoch(epoch)
            
            if (epoch+1) % self.config.save_gap:
                torch.save(self.model.state_dict(), f"{self.config.save_dir}/{epoch}.pth")
            if loss < loss_best:
                loss_best = loss
                torch.save(self.model.state_dict(), f"{self.config.save_dir}/best.pth")
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_accu = 0.0
        for batch, (data, gdth) in tqdm(enumerate(self.dataloader_train), desc=f"epoch {epoch}", ncols=100, total=len(self.dataloader_train)):
            data, gdth = data.unsqueeze(1).to(self.rt_device), gdth.float().to(self.rt_device)
            
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = th_F.cross_entropy(pred, gdth)
            loss.backward()
            self.optimizer.step()
            
            loss = loss.item() / self.config.batch_size * 1e3
            accu = pred.max(dim=1)[1].eq(gdth.max(dim=1)[1]).int().sum().item() / len(data)
            total_loss += loss
            total_accu += accu
            if (batch+1)%self.config.logd_gap == 0:
                tqdm.write(f"[train] loss: {loss:.3f}:, accu: {accu:.3f}")
                self.log_writer.add_scalar("train/loss", scalar_value=loss, global_step=epoch*len(self.dataloader_train)+batch)
                self.log_writer.add_scalar("train/accu", scalar_value=accu, global_step=epoch*len(self.dataloader_train)+batch)
        loss = total_loss / len(self.dataloader_train)
        accu = total_accu / len(self.dataloader_train)
        return loss, accu
    
    def test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_accu = 0.0
        with torch.no_grad():
            for data, gdth in self.dataloader_test:
                data, gdth = data.unsqueeze(1).to(self.rt_device), gdth.float().to(self.rt_device)
                pred = self.model(data)
                total_loss += th_F.cross_entropy(pred, gdth).item()
                total_accu += pred.max(dim=1)[1].eq(gdth.max(dim=1)[1]).int().sum().item() / len(data)
        loss = total_loss / len(self.dataloader_test) * 1e3
        accu = total_accu / len(self.dataloader_test)
        print(f"[test] loss: {loss:.3f}, accu: {accu:.3f}")
        self.log_writer.add_scalar(tag="test/loss", scalar_value=loss, global_step=(epoch+1)*len(self.dataloader_train))
        self.log_writer.add_scalar(tag="test/accu", scalar_value=accu, global_step=(epoch+1)*len(self.dataloader_train))
        
        return loss, accu

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    time_stmp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    mnist_raw = get_mnist_dataset()
    
    trainer = Trainer(
        dataset=myds.MyDataset(mnist_raw.data, mnist_raw.targets),
        config=dict({
            "ratio_train": 0.2,
            "ratio_eval": 0.1,
            "ratio_test": 0.7,
            "epoch": 10,
            "batch_size": 10,
            "lr": 1e-4,
            "logd_gap": 100,
            "save_gap": 10,
            "logd_dir": f"log/{time_stmp}",
            "save_dir": f"run/{time_stmp}"
        })
    )
    trainer.train()
    
