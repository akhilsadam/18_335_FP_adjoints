import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

# assuming f(t, u) is the derivative function we will integrate

# integrator is an object that has # of backward steps / adaptive integration etc.
# predefined along with the integration method
# - integrate(f, t, t_final, u-vars, options)

class AdjointSanity(torch.autograd.Function): # Does not work completely, testbed for the adjoint method
    
    @staticmethod
    def forward(ctx, integrator, f, t_initial, t_final, u, options={}):
        
        with torch.no_grad():
            u_final = integrator.integrate(f, t_initial, t_final, u, options)
        u_final.requires_grad_()
        
        ctx.integrator = integrator
        ctx.f = f
        ctx.options = options
        
        
        ctx.save_for_backward(u, t_initial, u_final, t_final)
        return u_final
    
    @staticmethod
    def backward(ctx, grad_output):
        print("BWD CALLED")
        u_initial, t_initial, u, t = ctx.saved_tensors
        
        options = ctx.options
        f = ctx.f
        integrator = ctx.integrator
        
        adjoint = grad_output
        dt = 0.1
        f_u = f(t, u)
        torch.autograd.backward(f_u, - dt * adjoint, retain_graph=True) 
  
        return None, adjoint, None, None, None, None



class SampleHold: # aka a left-Riemann sum
    def __init__(self, t):
        self.t = t
        
    def accumulate(self, f_u, adjoint, t):
        dt = torch.abs(self.t - t) # assuming backward integration
        # integrate -adjoint @ d(f_u)_dh over the interval
        torch.autograd.backward(f_u, dt * adjoint, retain_graph=True) 
        
        self.t = t

class BasicAdjointMethod(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, integrator, f, t_initial, t_final, u, options={}):

        with torch.no_grad():
            u_final = integrator.integrate(f, t_initial, t_final, u, options)
        u_final.requires_grad_()
        
        ctx.integrator = integrator
        ctx.f = f
        ctx.options = options
        
        ctx.save_for_backward(u, t_initial, u_final, t_final)
        return u_final
    
    @staticmethod
    def backward(ctx, grad_output):
        u_initial, t_initial, u, t = ctx.saved_tensors
                                        
        options = ctx.options
        f = ctx.f
        integrator = ctx.integrator
                                                                          
        adjoint = grad_output
        
        def _derivatives(t, args, accumulator=None):
            u, adjoint = args
            with torch.enable_grad():
                t.requires_grad_(True)
                u.requires_grad_(True)
                f_u = f(t, u)
                
                adjoint_f_grad_t, adjoint_f_grad_u = torch.autograd.grad(f_u, (t, u), - adjoint, allow_unused=True, retain_graph=True)
                
            with torch.no_grad():
                if accumulator is not None:
                    accumulator.accumulate(f_u, adjoint, t)
                    
                
                d_adjoint_dt = adjoint_f_grad_u # - adjoint * f_grad_u
                d_u_dt = f_u
                
            return d_u_dt, d_adjoint_dt

        
        def d_grad_dt(t, t_query, args):
            u, adjoint = args
            
            accumulator = SampleHold(t)
            derivatives = lambda t, args: _derivatives(t, args, accumulator=accumulator)
            
            u_prev, adjoint_prev = integrator.integrate(derivatives, t, t_query, [u, adjoint], options)
            updated_args = u_prev, adjoint_prev
                        
            return updated_args
        
        # TODO implement the proper outer integration?
        # below cheap sample-hold integration that skips outer loop
        u_hat_initial, adjoint_initial = d_grad_dt(t, t_initial, [u, adjoint])
        
        if 'consistency_loss' in options:
            options['consistency_loss'](u_hat_initial, u_initial).backward(retain_graph=True)
            
        adjoint_initial.requires_grad_(True)
  
        return None,  None, None, None, adjoint_initial, None
