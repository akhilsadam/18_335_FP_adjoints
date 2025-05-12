import torch
import torch.nn as nn
import torch.nn.functional as F

import models.act as _act

class Sum_Compressor(nn.Module):
    def __init__(self, in_channels, latent_channels, n_layers=1, act=nn.ReLU):
        super(Sum_Compressor, self).__init__()
        
        if isinstance(act, str):
            act = _act.act[act]
        
        pos = 2
        imd_chan = in_channels + latent_channels + pos
        
        self.encode = nn.Sequential(
            nn.Linear(in_channels+pos, imd_chan),
            act(),
            *([nn.Linear(imd_chan, imd_chan),
            act(),] * n_layers),
            nn.Linear(imd_chan, latent_channels),
        )
        
        self.decode = nn.Sequential(
            nn.Linear(latent_channels+pos, imd_chan),
            act(),
            *([nn.Linear(imd_chan, imd_chan),
            act(),] * n_layers),
            nn.Linear(imd_chan, in_channels),
        )
    
    def compressor(self, x):
        # B, C, H, W
        xpos = (torch.arange(x.shape[-2], device=x.device)[None, None, :, None] \
            .expand(x.shape[0], 1, x.shape[2], x.shape[3]) / x.shape[-2]) - 0.5
        ypos = (torch.arange(x.shape[-1], device=x.device)[None, None, None, :] \
            .expand(x.shape[0], 1, x.shape[2], x.shape[3]) / x.shape[-1]) - 0.5
        x = torch.cat([x, xpos, ypos], dim=1)
        xp = x.permute(0, 2, 3, 1) # B, H, W, C
        z = self.encode(xp) # B, H, W, C
        z_sum = z.mean(dim=(1, 2))
        return z_sum
    
    def decompressor(self, z_sum, xshape):
        xpos = (torch.arange(xshape[-2], device=z_sum.device)[None, :, None, None] \
            .expand(xshape[0], xshape[2], xshape[3], 1) / xshape[-2]) - 0.5
        ypos = (torch.arange(xshape[-1], device=z_sum.device)[None, None, :, None] \
            .expand(xshape[0], xshape[2], xshape[3], 1) / xshape[-1]) - 0.5
        z_expand = z_sum[:, None, None, :].expand(xshape[0], xshape[2], xshape[3], -1)
        z = torch.cat([z_expand, xpos, ypos], dim=-1)
        xp = self.decode(z) # B, H, W, C
        x = xp.permute(0, 3, 1, 2) # B, C, H, W
        return x
        
    def decompressor_w_pos(self, z_sum, z_pos, xshape):
        xpos = (torch.arange(xshape[-2], device=z_sum.device)[None, :, None, None] \
            .expand(xshape[0], xshape[2], xshape[3], 1) / xshape[-2]) - 0.5
        ypos = (torch.arange(xshape[-1], device=z_sum.device)[None, None, :, None] \
            .expand(xshape[0], xshape[2], xshape[3], 1) / xshape[-1]) - 0.5
        z_expand = z_sum[:, None, None, :].expand(xshape[0], xshape[2], xshape[3], -1)
        z_pos_expand = z_expand + z_pos
        z = torch.cat([z_pos_expand, xpos, ypos], dim=-1)
        xp = self.decode(z) # B, H, W, C
        x = xp.permute(0, 3, 1, 2) # B, C, H, W
        return x
        