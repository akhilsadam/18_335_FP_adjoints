import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.act as _act

# TBD.. need to discuss with others first

class WaveScatter(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def compressor(self, x):
        # B, C, H, W
        
    def decompressor(self, code):
        return val_decode(code, self.cxy, self.bitwidth) * 2 - self.shift
        
    def forward(self, x):
        return self.decompressor(self.compressor(x))
        