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

class Trainer:
    def __init__(
            self,
            model:torch.nn.Module,
            train_dataset: th_dataset.Dataset,
            test_dataset: th_dataset.Dataset,
            config: map
        ) -> None:
        # config map should contain:
        #   epoch
        #   batch_size
        #   train/eval/test ratio
        print("[========== model info ==========]")
        print(model)
        print("[========== dataset ==========]")
        print(train_dataset.__class__.__name__)
        self.config = AttributeMapping(config)
        
        self.rt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.rt_device)
        self.optimizer = th_optim.Adam(self.model.parameters(), self.config.lr)
        self.scheduler = th_optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)

        split_train, split_eval = torch.utils.data.random_split(
            train_dataset, [
                int(len(train_dataset)*self.config.ratio_train),
                int(len(train_dataset)*self.config.ratio_eval),
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
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.log_writer = tfx.SummaryWriter(self.config.logd_dir)
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def train(self):
        loss_best = float(1 << 31)
        for epoch in range(self.config.epoch):
            tran_loss, tran_accu = self.train_epoch(epoch)
            eval_loss, eval_accu = self.evaluate_epoch(epoch)
            test_loss, test_accu = self.test_epoch(epoch)
            self.scheduler.step()
            
            if (epoch+1) % self.config.save_gap:
                torch.save(self.model.state_dict(), f"{self.config.save_dir}/{epoch}.pth")
            if test_loss < loss_best:
                loss_best = test_loss
                torch.save(self.model.state_dict(), f"{self.config.save_dir}/best.pth")
            self.log_writer.add_scalars(
                "global/loss",
                {
                    "train_loss": tran_loss,
                    "eval_loss":  eval_loss,
                    "test_loss":  test_loss
                },
                epoch+1
            )
            self.log_writer.add_scalars(
                "global/accu",
                {
                    "train_accu": tran_accu,
                    "eval_accu" : eval_accu,
                    "test_accu" : test_accu
                },
                epoch+1
            )
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_accu = 0.0
        for batch, (data, gdth) in tqdm(enumerate(self.dataloader_train), desc=f"epoch {epoch}", ncols=100, total=len(self.dataloader_train)):
            data, gdth = data.to(self.rt_device), gdth.to(self.rt_device)
            
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = th_F.cross_entropy(pred, gdth)
            loss.backward()
            self.optimizer.step()
            
            loss = loss.item() / self.config.batch_size * 1e3 # expand to see detial variation
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
    
    def evaluate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_accu = 0.0
        with torch.no_grad():
            for data, gdth in self.dataloader_eval:
                data, gdth = data.to(self.rt_device), gdth.to(self.rt_device)
                pred = self.model(data)
                total_loss += th_F.cross_entropy(pred, gdth).item()
                total_accu += pred.max(dim=1)[1].eq(gdth.max(dim=1)[1]).int().sum().item() / len(data)
        loss = total_loss / len(self.dataloader_eval) * 1e3
        accu = total_accu / len(self.dataloader_eval)
        print(f"[eval] loss: {loss:.3f}, accu: {accu:.3f}")
        self.log_writer.add_scalar(tag="eval/loss", scalar_value=loss, global_step=(epoch+1)*len(self.dataloader_train))
        self.log_writer.add_scalar(tag="eval/accu", scalar_value=accu, global_step=(epoch+1)*len(self.dataloader_train))
        
        return loss, accu
    
    def test_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_accu = 0.0
        with torch.no_grad():
            for data, gdth in self.dataloader_test:
                data, gdth = data.to(self.rt_device), gdth.to(self.rt_device)
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
    time_stmp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dataset_name = "CIFAR10"
    model_name = "Conv2dCIFAR10"

    trainer = Trainer(
        train_dataset=myds.Inner(name=dataset_name, num_cls=10, root="data", train=True, download=True),
        test_dataset=myds.Inner(name=dataset_name, num_cls=10, root="data", train=False, download=True),
        model=mydl.MODEL[model_name](),
        config=dict({
            "ratio_train": 0.4,
            "ratio_eval": 0.6,
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
