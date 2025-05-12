import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipConnection(nn.Module):
    def __init__(self, *args):
        super(SkipConnection, self).__init__()
        self.m = nn.Sequential(*args)
        
        self.weight = nn.Parameter(torch.tensor([1e-3])) # trainable parameter
        
    def forward(self, x):
        return x + self.m(x) * self.weight
    
class ScaleLayer(nn.Module):

   def __init__(self, scale=1e-3):
       super().__init__()
       self.scale = scale

   def forward(self, input):
       return input * self.scale