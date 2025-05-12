import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)
    
class AddSine(nn.Module):
    def __init__(self):
        super(AddSine, self).__init__()
        
        self.a = nn.Parameter(torch.tensor(0.001))
        self.w = nn.Parameter(torch.tensor(30.0))
    def forward(self, x):
        return x + self.a * torch.sin(self.w * x)