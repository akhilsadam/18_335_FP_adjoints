import numpy as np

# 1D viscous Burgers' equation

# Define the domain
x = np.linspace(0, 1.0, 100) # some reason <100 is stable, 101 is not
Re = 1000
T = 6.0
save_time = 0.01

# Define the initial condition
t0 = np.exp(Re/8)
u0 = x / (1 + np.sqrt(1/t0) *(np.exp(Re * x**2 / 4)))

# solve with RK45
from scipy.integrate import solve_ivp

def f(t, u):
    du_dx = np.gradient(u, x, edge_order=2)
    d2u_d2x = np.gradient(du_dx, x, edge_order=2)
    
    du_dt = -u * du_dx + 1/Re * d2u_d2x

    return du_dt

sol = solve_ivp(f, [0, T], u0, method='RK45', t_eval=np.arange(0, T, save_time), rtol=1e-6)

# save the solution into three sets of train, infer, and test
y =  sol.y.T
print(y.shape)
nt = y.shape[1]
np.save('train.npy', y[:nt//3, None, :])
np.save('test.npy', y[nt//3:2*nt//3, None, :])
np.save('infer.npy', y[2*nt//3:, None, :])

import matplotlib.pyplot as plt
plt.imshow(y, aspect='auto')
plt.savefig('solution.png')