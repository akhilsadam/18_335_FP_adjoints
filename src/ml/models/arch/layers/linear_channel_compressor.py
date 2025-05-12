import torch
import torch.nn as nn
import torch.nn.functional as F

import models.act as _act

class Linear_Channel_Compressor(nn.Module):
    def __init__(self, in_chan, out_chan, n_layers=1, act=nn.ReLU):
        super(Linear_Channel_Compressor, self).__init__()
        
        if isinstance(act, str):
            act = _act.act[act]
        
        pos = 2
        imd_chan = in_chan + out_chan + pos
        
        self.process = nn.Sequential(
            nn.Linear(in_chan+pos, imd_chan),
            act(),
            *([nn.Linear(imd_chan, imd_chan),
            act(),] * n_layers),
            nn.Linear(imd_chan, out_chan),
        )
    
    def _compressor_no_sum(self, x):
        # B, C, H, W
        xpos = (torch.arange(x.shape[-2], device=x.device)[None, None, :, None] \
            .expand(x.shape[0], 1, x.shape[2], x.shape[3]) / x.shape[-2]) - 0.5
        ypos = (torch.arange(x.shape[-1], device=x.device)[None, None, None, :] \
            .expand(x.shape[0], 1, x.shape[2], x.shape[3]) / x.shape[-1]) - 0.5
        x = torch.cat([x, xpos, ypos], dim=1)
        xp = x.permute(0, 2, 3, 1) # B, H, W, C
        z = self.process(xp) # B, H, W, C
        return z
    
    def compressor(self, x):
        # B, C, H, W
        z = self._compressor_no_sum(x) # B, H, W, C
        z_ = z.permute(0, 3, 1, 2) # B, C, H, W
        return z_
        
    def forward(self, x):
        return self.compressor(x)
        