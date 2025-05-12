import torch
def pdf_x(x):
    f = x**2 # wavefunction density
    f = f / torch.sum(f, dim=(1,2,3), keepdim=True)
    h = -torch.sum(f * torch.log(f + 1e-8), dim=(1,2,3)) # wavefunction entropy # B, L, H, W -> B
    d_f = torch.exp(-h) # wavefunction pdf based on thermodynamic entropy
    pdf = d_f / torch.sum(d_f)
    return pdf

def pdf_y(x):
    f = x**2 # wavefunction density
    f = f / torch.sum(f, dim=(1,2,3), keepdim=True)
    h = -torch.sum(f * torch.log(f + 1e-8), dim=(1)) # wavefunction entropy # B, L -> B
    d_f = torch.exp(-h) # wavefunction pdf based on thermodynamic entropy
    pdf = d_f / torch.sum(d_f)
    return pdf

def latent_entropy(x):
    f = x**2 # wavefunction density
    f = f / torch.sum(f, dim=(1), keepdim=True) # normalize distribution B, L 
    h = - torch.sum(f * torch.log(f + 1e-8), dim=(1)) # wavefunction entropy # B, L -> B
    return torch.mean(h)

def rev_kl(x, y):
    # x | y
    px = pdf_x(x)
    py = pdf_y(y)
    return torch.sum(px * (torch.log(px) / (py + 1e-8)))