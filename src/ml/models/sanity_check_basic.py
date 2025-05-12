import torch
import torch.nn as nn

from parameters.parameter import DefaultParameters
from models.interface import NetInterface
from lightning.pytorch import LightningModule

from models.adjoint.adjoint import BasicAdjointMethod
from models.adjoint.integrator import BasicIntegrator

class Exp(LightningModule):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, i):
        with torch.no_grad():
            result = i.exp()
            self.saved_tensors = result
        return result
    
    def backward(self, grad_output):
        with torch.no_grad():
            result = self.saved_tensors
        return grad_output * result
# Use it by calling the apply method:

# class Exp(torch.autograd.Function):
    
#     @staticmethod
#     def forward(ctx, i):
#         with torch.no_grad():
#             result = i.exp()
#             ctx.save_for_backward(result)
#         return result
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         with torch.no_grad():
#             result, = ctx.saved_tensors
#         return grad_output * result

class Parameters(DefaultParameters): # default parameters for the model
    def __init__(self):
        super().__init__()
        self.lr = 0.025
        self.it = 4000
        self.batch_size = 12
        self.crit = nn.MSELoss()
        self.vcrit = nn.MSELoss()
        
class Network(NetInterface, LightningModule):
    def __init__(self, param: DefaultParameters):
        """method used to define our model parameters"""
        super().__init__(param)

        self.weight = nn.Parameter(torch.randn(1)) 
        self.dx = 0.1
        self.nu = nn.Parameter(torch.randn(1)) # trainable parameters
        self.time = torch.randn(1) * 0.0
        
        
        self.integrator = BasicIntegrator(method='euler', tol=1e-3, n_steps=100)
        
        self.expm = Exp() # test
        
        self.save_hyperparameters() # save hyper-parameters to self.hparams (auto-logged by W&B)

    def f(self, t, u, options):
        
        du_dx = torch.gradient(u, edge_order=2, dim=-1)[0] / self.dx
        d2u_d2x = torch.gradient(du_dx, edge_order=2, dim=-1)[0] / self.dx
        
        du_dt = self.weight * ( -u * du_dx + self.nu * d2u_d2x)
        
        return du_dt

    def forward(self, u):
        # batch, seq, channels, x [y,z]
        self.time = self.time.to(u.device)
        
        # u_int = self.weight * torch.cos(self.nu) * u # works
        # u_int = self.weight * torch.cos(self.nu) * Exp.apply(u)  # fails - why?
        u_int = self.weight * torch.cos(self.nu) * self.expm(u)  #  works somehow..
        
        #BasicAdjointMethod.apply(self.integrator, self.f, self.time, 1.0+self.time, u)
        # assert u_int.requires_grad
        print(u_int.requires_grad)

        return u_int