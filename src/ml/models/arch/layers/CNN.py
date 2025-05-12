import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.connections.skip import SkipConnection
# 
# import models.act as _act

# def init_conv_near_identity(m):
#     if isinstance(m, nn.Conv2d):
#         with torch.no_grad():
#             # Initialize weights to identity
#             m.weight *= 0.0
#             m.weight[:,:,m.weight.shape[2]//2, m.weight.shape[3]//2] = 1.0/m.weight.shape[0]

#             # Add a small amount of noise
#             m.weight.data.add_(torch.randn_like(m.weight) * 0.01)

#             # Initialize bias to zero
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

# class CNN(nn.Module):
#     def __init__(self, in_channels, latent_channels, kernel_size, n_layers=2):
#         super().__init__()
                
#         self.n_layers = n_layers
        
#         assert kernel_size % 2 == 1, "Kernel size must be odd"
#         assert latent_channels % (self.n_layers) == 0, "Latent channels must be divisible by n_layers"
#         lat_channels = latent_channels // self.n_layers
        
#         self.convs = nn.ModuleList([nn.Conv2d(in_channels, lat_channels, kernel_size, padding='same') for _ in range(self.n_layers)])
#         self.deconvs = nn.ModuleList([nn.Conv2d(lat_channels, in_channels, kernel_size, padding='same') for _ in range(self.n_layers-1)])
        
#     def forward(self, x):
#         out = []
#         z = self.convs[0](x)
#         out.append(z)
#         for conv, deconv in zip(self.convs[1:], self.deconvs):
#             x = x - deconv(z) # finer scale residual
#             z = conv(x)
#             out.append(z)
         
#         return torch.cat(out, dim=1) # BCHW

class CNN(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size, n_layers=2):
        super().__init__()
                
        self.n_layers = n_layers
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, latent_channels, kernel_size, padding='same') for _ in range(self.n_layers)])
        self.deconvs = nn.ModuleList([nn.Conv2d(latent_channels, in_channels, kernel_size, padding='same') for _ in range(self.n_layers-1)])
        self.w = nn.Parameter(torch.tensor(0.001))
        
    def forward(self, x):
        out = []
        z = self.convs[0](x)
        out.append(z)
        for conv, deconv in zip(self.convs[1:], self.deconvs):
            x = x - self.w * deconv(z)
            z = conv(x)
        return z