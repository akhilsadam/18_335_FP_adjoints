import numpy as np
import jpcm
import matplotlib.colors as mcolors
from einops import rearrange
from PIL import Image
import os

from math import sqrt,cos,sin,radians
class RGBRotate(object):
    def __init__(self):
        self.matrix = [[1,0,0],[0,1,0],[0,0,1]]

    def set_hue_rotation(self, degrees):
        cosA = cos(radians(degrees))
        sinA = sin(radians(degrees))
        self.matrix[0][0] = cosA + (1.0 - cosA) / 3.0
        self.matrix[0][1] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[0][2] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][0] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][1] = cosA + 1./3.*(1.0 - cosA)
        self.matrix[1][2] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][0] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][1] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[2][2] = cosA + 1./3. * (1.0 - cosA)

    def apply(self, rgb):
        r = rgb[:,0]
        g = rgb[:,1]
        b = rgb[:,2]
        rx = r * self.matrix[0][0] + g * self.matrix[0][1] + b * self.matrix[0][2]
        gx = r * self.matrix[1][0] + g * self.matrix[1][1] + b * self.matrix[1][2]
        bx = r * self.matrix[2][0] + g * self.matrix[2][1] + b * self.matrix[2][2]
        return np.stack([np.clip(rx,0,1), np.clip(gx,0,1), np.clip(bx,0,1), 1.0+0.0*bx],axis=-1)
hue = RGBRotate()
hue.set_hue_rotation(180)
scmap = jpcm.get('sky')(np.linspace(-0.2, 1.0, 128))
rmap = scmap[::-1]
rmap = hue.apply(rmap)
rmap[:,3] = 1
cmap = mcolors.LinearSegmentedColormap.from_list('div',(np.vstack((scmap, rmap))))

def scan_plot(path, name, labels=None, data=None):

    if data is None:
        labels = ['train','valid','infer']
        data = []
        for i, label in enumerate(labels):
            scan = np.load(os.path.join(path,f'{label}_scans.npz'))
            frames = scan['arr_0']
            data.append(frames)
            
    assert labels is not None, 'labels must be provided'
    assert data is not None, 'data must be provided'
    assert len(labels) == len(data), 'labels and data must be the same length'
            
    n_latent, n_step, n_c, _,_ = data[0].shape
    ds = []
    for i, label in enumerate(labels):
        frames = data[i] 
        du_dx, du_dz = np.gradient(frames[:,:,1,...], axis=(-2,-1))
        dw_dx, dw_dz = np.gradient(frames[:,:,0,...], axis=(-2,-1))
        w = dw_dx + du_dz
        frames = np.concatenate([frames, w[:,:,None,...]], axis=2)
        ds.append(frames)
        
    ds = np.stack(ds,axis=0)
    ds = rearrange(ds,'t l s c h w -> l (c h) (t s w)')
    for l in range(n_latent):  
        d = ds[l]
        d = (d - d.min()) / (d.max() - d.min())
        out = Image.fromarray((cmap(d) * 255).astype(np.uint8))
        out.save(os.path.join(path, f'{name}_scan_{l}.png'))