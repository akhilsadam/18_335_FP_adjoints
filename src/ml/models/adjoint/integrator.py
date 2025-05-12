import torch
import torch.nn as nn

def euler(f, t, t_final, u_vars, options, n_steps): 
    dt = t_final - t
    h = dt / n_steps

    for _ in range(n_steps):
        du_s = f(t, u_vars)
        
        for j in range(len(u_vars)):
            u_vars[j] = u_vars[j] + h * du_s[j]
        
        t = t + h

    return u_vars

def rk4(f, t, t_final, n_steps, u):  # TODO fix (generated code)
    dt = t_final - t
    h = dt / n_steps

    for i in range(n_steps):
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)

        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

    return u

class BasicIntegrator(nn.Module): # that is, not differentiable
    def __init__(self, adjoint_method, method='euler', tol=1e-3, n_steps=10):
        super().__init__()
        self.method = method
        self.tol = tol
        self.n_steps = n_steps
        self.adjoint_method = adjoint_method
        
    def forward(self, f, t_initial, t_final, u, options={}):
        u.requires_grad_()
        return self.adjoint_method.apply(self, f, t_initial, t_final, u, options)

    def integrate(self, f, t, t_final, u_vars, options={}):
        return eval(self.method)(f, t, t_final, u_vars, options, self.n_steps)



