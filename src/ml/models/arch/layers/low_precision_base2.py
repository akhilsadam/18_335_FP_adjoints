import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.act as _act

### Actually not trivial as Pytorch does not support binary addition...without using the bitshift operator
### Can we even backpropagate through this?

def pos_encode(cpos, xpos, ypos, C, H, W):
    _c = cpos
    _x = xpos * C
    _y = ypos * C * H
    return (_c + _x + _y)

def val_encode(x, cxy, bitwidth):
    _cxy = (bitwidth * cxy)
    
    # make this into a vector of bitwidth
    bw = 2**torch.arange(bitwidth, device=x.device)[None, None, None, None,:]
    xq = torch.round(x * (2**bitwidth - 1) / bw).to(torch.int32)
    # now we have b c h w bitwidth and it is binary!
    
    # quantized x in [0,2**bitwidth-1], in linear integer space.
    zq = torch.bitwise_left_shift(xq,_cxy) # integer windows of [0,(2**bitwidth-1)/(2**bitwidth)] starting from cxy
    # now change base for summing
    zq2 = 10**zq

    # now combine everything
    return torch.sum(x_qb2,dim=(1,2,3), keepdim=True)

def val_decode(code, cxy, bitwidth):
    _cxy = (bitwidth * cxy)
    x_b2 = torch.clamp(torch.round(code - _cxy), 0, bitwidth)
    x = (2.0**x_b2) / (2**bitwidth - 1)
    return x

class IntegerQuantization(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.shift = nn.Parameter(torch.tensor(1.0))
    
    def compressor(self, x):
        # B, C, H, W
        # assumes normalized in [-1,1]
        # shift normalization to [0,1]
        x = (x + 1) / 2
        # we have 15 integers (ie log 10) and input around 400x400
        # let's give 3+1 bits / bitwidth to each location
        # so we have values in [0, 2^(bitwidth-1)-1] and a sign
        # so that is 2^53 values / 2^(bitwidth) = about 2^49 values we can store...
        if 'bitwidth' not in self.__dict__:
            self.dims = x.shape[1:]
            n = x.shape[1] * x.shape[2] * x.shape[3]
            # reversing computation to find bitwidth; 24 for single precision
            bitwidth = int(24 - (math.log(n) / math.log(2)))
            # we have to store the bitwidth
            self.bitwidth = bitwidth
            
            # cxy positional encoding
            xpos = torch.arange(x.shape[-2], device=x.device)[None, None, :, None]
            ypos = torch.arange(x.shape[-1], device=x.device)[None, None, None, :]
            cpos = torch.arange(x.shape[1], device=x.device)[None, :, None, None]
            
            self.cxy = pos_encode(cpos, xpos, ypos, x.shape[1], x.shape[2], x.shape[3])
            print('CXY:',self.cxy)
            print('Bitwidth:',self.bitwidth)
        
        return val_encode(x, self.cxy, self.bitwidth)
    
    def decompressor(self, code):
        return val_decode(code, self.cxy, self.bitwidth) * 2 - self.shift
        
    def forward(self, x):
        return self.decompressor(self.compressor(x))
        