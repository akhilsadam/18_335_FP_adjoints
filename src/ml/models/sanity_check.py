import torch
import torch.nn as nn

from parameters.parameter import DefaultParameters
from models.interface import NetInterface
from lightning.pytorch import LightningModule

from models.adjoint.adjoint import BasicAdjointMethod, AdjointSanity
from models.adjoint.integrator import BasicIntegrator

class Parameters(DefaultParameters): # default parameters for the model
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.it = 4000
        self.batch_size = 8
        self.crit = nn.MSELoss()
        self.vcrit = nn.MSELoss()
        
class Network(NetInterface, LightningModule):
    def __init__(self, param: DefaultParameters):
        """method used to define our model parameters"""
        super().__init__(param)

        self.dx = 1/99.0 # spatial resolution
        self.nu = nn.Parameter(torch.randn(1) * 0.01) # trainable parameters
        self.weight = nn.Parameter(torch.randn(1) * 0.01) # trainable parameters
        
        self.time = torch.randn(1) * 0.0

        self.adjoint = BasicIntegrator(BasicAdjointMethod, method='euler', tol=1e-3, n_steps=10)

        self.save_hyperparameters() # save hyper-parameters to self.hparams (auto-logged by W&B)

    def f(self, t, u):
        
        du_dx = torch.gradient(u, edge_order=2, dim=-1)[0] / self.dx
        d2u_d2x = torch.gradient(du_dx, edge_order=2, dim=-1)[0] / self.dx
        
        du_dt = ( -u * du_dx * self.weight + self.nu * d2u_d2x)
        
        return du_dt

    def forward(self, u):
        # batch, seq, channels, x [y,z]
        self.time = self.time.to(u.device)
        
        u_int = self.adjoint(self.f, self.time, 0.01 + self.time, u[:,-1:,...]) # single step forward

        print(self.weight, self.nu)

        return u_int
    