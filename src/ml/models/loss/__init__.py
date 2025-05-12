from .EMD import L2_Kernel_EMD
import torch.nn as nn

loss_types={
    'L2_Kernel_EMD': lambda kwargs: L2_Kernel_EMD(**kwargs),
    'mse': lambda kwargs: nn.MSELoss(),
}