import torch
from lightning.pytorch import LightningModule

class L2_Kernel_EMD(LightningModule):
    def __init__(self, kernel_size, pixel_dist=.25):
        super().__init__()

        self.unfold = lambda x :  x.unfold(-2, kernel_size, 1).unfold(-2, kernel_size, 1)
        # self.mse = torch.nn.MSELoss()
        # BCHW -> BC(#h)Wh -> BC(#h)(#w)hw

         # Get center patch indices
        center_h = kernel_size // 2
        center_w = kernel_size // 2
        self.center = lambda x: x[...,center_h, center_w][...,None,None]
        
        # x, y from center
        _x = torch.arange(kernel_size, dtype=torch.float32) - center_h
        _y = torch.arange(kernel_size, dtype=torch.float32) - center_w
        self.register_buffer("dx", pixel_dist * _x[:,None].expand(kernel_size, kernel_size)[None, None, None, None, ...])
        self.register_buffer("dy", pixel_dist * _y[None,:].expand(kernel_size, kernel_size)[None, None, None, None, ...])
        # BC(#h)(#w)hw

    def forward(self, yhat, y):
        # yhat, y: B, C, H, W
        # B, C, H, W = y.shape
        
        if self.dx.device != y.device:
            self.dx = self.dx.to(y.device)
            self.dy = self.dy.to(y.device)
            # workaround for pytorch lightning bug
        

        yhat_unf = self.unfold(yhat)
        y_unf = self.unfold(y) # B, C*kernel_size**2, H*W
        
        # kernel centers
        y_cen = self.center(y_unf)
        
        # z error
        # print(yhat_unf.shape, y_cen.shape)
        dz = (yhat_unf - y_cen)
        # x error
        l2 = self.dx**2 + self.dy**2 + dz**2
        
        # reduce to find min error
        l2_min = torch.amin(l2, dim=(-2,-1)) # B, C, (#h) (#w) = B, C, H, W
        
        # sum reduce to make MSE
        mse = torch.mean(l2_min) # scalar
        
        return mse #+ 0.05 * self.mse(yhat, y)
        
        
        