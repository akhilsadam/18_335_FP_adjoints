import torch
import torch.nn as nn
from torch.optim import Adam
from lightning.pytorch import LightningModule

from parameters.parameter import DefaultParameters
import models.loss as losses

class NetInterface(LightningModule):
    def __init__(self, param: DefaultParameters):
        """method used to define our model parameters"""
        super().__init__()
        self.param = param
        
        # find loss
        if isinstance(param.crit,tuple):
            self.param.crit = losses.loss_types[param.crit[0]](param.crit[1])
        if isinstance(param.vcrit,tuple):
            self.param.vcrit = losses.loss_types[param.vcrit[0]](param.vcrit[1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.param.crit(y_hat, y) / self.param.vcrit(y, torch.mean(y, dim=tuple(list(range(2,len(x.shape)))), keepdim=True))
        # Log loss and metric
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat.shape, y.shape, x.shape)
        loss = self.param.vcrit(y_hat, y) / self.param.vcrit(y, torch.mean(y, dim=tuple(list(range(2,len(x.shape)))), keepdim=True))
        # Log loss and metric
        self.log("valid_loss", loss.item(), on_step=True, on_epoch=False)
        return torch.stack([y_hat - y, y], dim=1)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return torch.stack([y_hat - y, y], dim=1)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.param.lr)
    