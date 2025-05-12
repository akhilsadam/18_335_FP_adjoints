import holoviews as hv
import numpy as np
import panel as pn
import bokeh
from bokeh.resources import INLINE
import matplotlib.colors as mcolors
from einops import rearrange
import os

hv.extension('bokeh') 
hv.extension('matplotlib')
hv.extension('plotly')

def seq_plot(path, label, res_lat=None):

    if res_lat is None:
        res_lat = np.load(os.path.join(path,f'{label}_res_latents.npz'))['arr_0']

    if res_lat.shape[2] >= 3:
        def _curve(st, k):
            dat = np.concatenate([res_lat[st:st+k, :, :3],np.arange(res_lat.shape[1])[None,::-1,None].repeat(k,axis=0)],axis=2)
            dat = rearrange(dat, 'b t c -> (b t) c')
            return hv.Scatter3D({('x', 'y', 'z','w'): dat}, kdims=['x','y','z'], vdims=['w'])
        
    elif res_lat.shape[2] == 2:
        def _curve(st, k):
            dat = np.concatenate([res_lat[st:st+k, :, :2],np.arange(res_lat.shape[1])[None,::-1,None].repeat(k,axis=0)],axis=2)
            dat = rearrange(dat, 'b t c -> (b t) c')
            return hv.Scatter({('x', 'y','w'): dat}, kdims=['x','y'], vdims=['w'])
    else: 
        # 1D
        print('1D; not implemented')
        return
    
    n = res_lat.shape[0]

    # sts = list(range(0,n,n//10))
    # ks = [1, 5, n]
    curve_func = lambda st, k : _curve(st,min(k,n-st)).opts(cmap='fire_r', color='w', colorbar=True, title=f'{label} latent sequences')
    dmap = hv.DynamicMap(curve_func, kdims=['start', 'count'])
    dmap = dmap.redim.values(start=list(range(0,n,1)), count=(1,2,3,4,5,n))
    # panel_object = pn.pane.HoloViews(hmap)
    hv.save(dmap, os.path.join(path, f'{label}_latent_seq.html'), resources=INLINE)