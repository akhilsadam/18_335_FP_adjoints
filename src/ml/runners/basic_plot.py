import os
import ffmpeg
import numpy as np
import jpcm
from tqdm import tqdm
import matplotlib.colors as mcolors
from einops import rearrange

# scmap = jpcm.get('sky')(np.linspace(-0.2, 1., 128))
# fcmap = jpcm.get('sunburst')(np.linspace(1., -0.2, 128))
# cmap = mcolors.LinearSegmentedColormap.from_list('div',np.vstack((scmap, fcmap)))

remake = True #False

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



names = ['valid','infer']

directory = 'run/'

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def mp4(filename, d, fps=2):
    """Saves a NumPy array of frames as an MP4 video."""
 
    if d.shape[0] > 4000:
        d = d[1000:3000,...]
 
 
    frames = np.stack([ d[:,0,...], d[:,0,...] + d[:,1,...],d[:,1,...]], axis=1).astype(np.float32) # x_err, xhat, x
 
    # add vorticity (1/2, just curl)
    # yx-xy
    # ground truth
    # g = np.gradient(frames[:,1:], axis=(-2,-1)) # B, type=xhat,x, C, H, W, ; d/dx, d/dy
    # w = g[1][:,:,1,...] - g[0][:,:,0,...] # B, type=xhat,x, H, W
 
    du_dx, du_dz = np.gradient(frames[:,1:,1,...], axis=(-2,-1))
    dw_dx, dw_dz = np.gradient(frames[:,1:,0,...], axis=(-2,-1))
    w = dw_dx + du_dz

    w_frames = np.stack([w[:,0,...] - w[:,1,...],w[:,0,...],w[:,1,...]], axis=1)
    
    frames = np.concatenate([frames, w_frames[:,:,None,...]], axis=2)
    print(frames.shape)
    r = np.maximum(np.max(frames[:,-1], axis=(0,2,3)),-np.min(frames[:,-1],axis=(0,2,3)))[None,None,:,None,None] # B, Y, C, H, W
    nframes = frames / (2*r) + 0.5
    nframes = rearrange(nframes,'b y c h w -> b (c h) (y w)')
    print(np.max(nframes), np.min(nframes))
    frames = cmap(nframes)[...,:3]
    frames *= 255 / np.max(frames)
    n, height, width, channels = frames.shape
    print(frames.shape)
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(filename, pix_fmt='yuv420p', vcodec='libx264', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


if __name__ == '__main__':
    print('scan directory')
    gen = fast_scandir(directory)
    for path in tqdm(gen):
        fname = os.path.join(path,'valid.mp4')
        qname = os.path.join(path,'infer.npz')
        if (not os.path.exists(fname) and os.path.exists(qname)) \
              or (os.path.exists(qname) and remake):
            print(path)
            try:
                data = [np.load(os.path.join(path,f"{x}.npz"))['arr_0'] for x in names]    
                for name, d in zip(names, data):
                    fname = os.path.join(path,f'{name}.mp4')
                    mp4(fname, d)
            except Exception as e:
                print(e)

# run this script with 
# conda activate ./sw
# source .venv/bin/activate
# python3 src/ml/runners/basic_plot.py