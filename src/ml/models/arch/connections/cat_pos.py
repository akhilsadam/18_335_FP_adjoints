import torch
import torch.nn as nn

class Cat(nn.Module):
    def __init__(self, dim=1):
        super(ConcatModule, self).__init__()
        self.dim = dim

    def forward(self, x, external_tensor):
        return torch.cat((x, external_tensor), dim=self.dim)