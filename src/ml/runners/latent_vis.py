import os
import importlib
import importlib.machinery
import logging

import torch
torch.set_float32_matmul_precision('high') # can do medium as well

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from einops import rearrange

import numpy as np
import wandb


from pykeops.torch import LazyTensor

seed_everything(1)
from iox.load import loaders as _loaders
from utilities.repo import understand_env
# from utilities.gpulog import GPU_util # TODO

labels = ['train', 'valid', 'infer']

def KNN(x, k=3):
    G_i = LazyTensor(x[:, None, :])  # (M**2, 1, v-dim)
    X_j = LazyTensor(x[None, :, :])  # (1, N, v-dim)
    D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(k, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    return indKNN

def intrinsic(x, tag, label):
    indKNN = KNN(x, k=3)
    r2 = ((x[indKNN[:, 2]] - x) ** 2).sum(-1) # (M**2)
    r1 = ((x[indKNN[:, 1]] - x) ** 2).sum(-1) # (M**2)
    mu = r2 / r1
    
    # print(mu)
    # print(r2)
    # print(r1)    
    
    pdf, bin_edges = torch.histogram(mu, bins=25, density=True, range=(1, torch.amax(mu)))
    pdf = pdf / torch.sum(pdf)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    cdf = torch.cumsum(pdf, dim=0)
    # fit to 1-x^-d
    # log(1-cdf) is -d log(x)
    # d = - log(1-cdf) / log(x)
    # print(cdf)
    # print(bin_centers)
    d = - torch.log(1-cdf) / torch.log(bin_centers)
    d = d[~torch.isnan(d)]
    d = d[~torch.isinf(d)]
    print(d)
    
    ud = torch.mean(d)
    sd = torch.std(d)
    n_samples = d.shape[0]
    
    print(f'{label}_{tag}: {ud.item()} with std {sd.item()}; {n_samples} samples')

    f_val = 1 - bin_centers ** -ud
    
    # wbl.experiment.log_table(f'intrinsic_{tag}',
    #                          columns=None, data=None, dataframe=None, step=None)

    
    return (ud, sd), {
            
     f'intrinsic_{tag}' : 
         [
             wandb.plot.line_series,
            {
                        'xs':[bin_centers.cpu().numpy().tolist(),bin_centers.cpu().numpy().tolist()],
                        'ys':[cdf.cpu().numpy().tolist(),f_val.cpu().numpy().tolist()],
                        'keys':[f"{label}_{tag}", f"{label}_{tag}_fit_d_{ud.item():.2f}"],
                        'title':f"Intrinsic Dimensionality - {tag}",
                        'xname':"r2 / r1",
            }
         ]
    }
     
def run(config, py_logger):
    run_p = config['run_parameters']

    version, task, save_path = understand_env()
    fname = '-'.join(config['task_path'].split('/')[1:]) + f"-{config['run_id']}"
    
    # py_logger.info(f'Logging to wandb with {fname}')
    
    # which gpus are used?
    cvs = os.environ['CUDA_VISIBLE_DEVICES']
    cvs = [int(c) for c in cvs.split(',')] # convert to list of ints
    config.update({'which_gpu': cvs})
    
    # # os.makedirs(os.path.join(save_path,"wandb/"), exist_ok=True)'
    # wandb_logger = WLogger(project=config['project_name'], name=fname,
    #                 version=fname, config=config)
    
    run = wandb.init(project=config['project_name'], name=fname, config=config, dir=os.fspath('.'))
    
    # save_dir=save_path) # has issues with sync...
    # py_logger.info(f'wandb configured')
    
    tester = Trainer(accelerator="gpu", devices=1)
    
    model = importlib.import_module(config['run_parameters']['model'])
        
    if run_p.get('load_from', False):
        try:
            batch_size = config['run_parameters']['model_parameters'].get('batch_size', 1)
        except:
            batch_size = 1        
            
        l_kwargs = {'batch_size':batch_size,**config['run_parameters'].get('loader_parameters', {})}
        loaders, shapes, datasets = _loaders(config['data'], **l_kwargs)
        
        with tester.init_module(empty_init=True):
            configfile = os.path.join(os.path.dirname(run_p['load_from']),'../','system_parameters.py')
            py_logger.info(f'loading model from {configfile}')
            loader = importlib.machinery.SourceFileLoader('config', configfile)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            _config = importlib.util.module_from_spec(spec)
            loader.exec_module(_config)
            
            load_config = _config.system
            
            params = load_config.get('actions')[0].get('tasks')[0].get('run_parameters').get('model_parameters')
            py_logger.info(f'params: {params}')
            
            param = model.Parameters().update(**params)
            param.set_shapes(shapes)
            
            model = model.Network.load_from_checkpoint(run_p['load_from'], param=param)

        
        py_logger.info(f'loaded model from {run_p["load_from"]}')
    else:
        raise ValueError('load_from must be specified')
    
    model.eval() # may be redundant    

    def compute(model, function):
        model.predict_step = function
        outs = tester.predict(model, dataloaders=loaders)
        return [torch.concatenate(vs, dim=0) for vs in outs]
    
    def for_all(x, function):
        outs = []
        argsall = []
        for i, data in enumerate(x):
            y,args = function(data, labels[i])
            outs.append(y)
            argsall.append(args)
        
        for k in argsall[0].keys():
            func = argsall[0][k][0] # function
            fagl = []
            
            for i, carg in enumerate(argsall):
                if k not in carg:
                    continue
                fargs = carg[k][1]
                fagl.append(fargs) # list of function args in various dicts
               
            p = fagl[0] 
            for i in fagl[0].keys():
                if isinstance(fagl[0][i],list):
                    qi = []
                    for a in fagl:
                        qi.extend(a[i])
                    p[i] = qi
                
            run.log({k: func(**p)})
        
        return outs
    
    def save(name, data):
        for i, label in enumerate(labels):
            # print(data[i])
            print(data[i].shape)
            np.savez(os.path.join(save_path, f'{label}_{name}'), data[i])
                
    py_logger.info('computing data intrinsic dimension...')
    data_dims = for_all(datasets, lambda i,l: intrinsic(i.reshape((i.shape[0],-1)), 'data', l))
    py_logger.info(f'data_dims: {data_dims}')
    
    py_logger.info('computing latent vectors...')
    latents = compute(model, model.latent_vectors)
    save('latents', latents)
    
    py_logger.info('computing latent intrinsic dimension...')
    latent_dims = for_all(latents, lambda i,l: intrinsic(i.reshape((i.shape[0],-1)), 'latent', l))
    py_logger.info(f'latent_dims: {latent_dims}')
    
    py_logger.info('computing reco vectors...')
    recos = compute(model, model.reco_vectors)
    save('recos', recos)
    
    py_logger.info('computing reconstruction intrinsic dimension...')
    reco_dims = for_all(recos, lambda i,l: intrinsic(i.reshape((i.shape[0],-1)), 'reconstruction', l))
    py_logger.info(f'reco_dims: {reco_dims}')

    py_logger.info('computing error intrinsic dimension...')
    err_dims = for_all(zip(recos, datasets), lambda i,l: intrinsic(i[0].reshape((i[0].shape[0],-1)) - i[1].reshape((i[1].shape[0],-1)), 'errors', l))
    py_logger.info(f'err_dims: {err_dims}')
    
    py_logger.info('identifying manifold...by training FCN autoencoder')
    # train FCN autoencoder on latent vectors
    intrinsic_dim = int(max([d[0].item() for d in latent_dims]) + 1.0)
    py_logger.info(f'using intrinsic_dim: {intrinsic_dim}')
    
    from models import latent_net
    lparams = {'in_dim': latents[0].shape[1], 'intrinsic_dim': intrinsic_dim, 'lr': 1e-3, 'epochs': 60, 'batch_size': 8}
    latent_loaders = [DataLoader(data, shuffle=shuffle, pin_memory=True, **l_kwargs) for data, shuffle in zip(latents, [True, False, False])]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_path,"ckpt/"),
            every_n_train_steps=config['save_frequency'],
        ),
    ]
    latent_model = latent_net.Network(latent_net.Parameters().update(**lparams))
    wandb_logger = WandbLogger(name=f'{fname}-latent', project=config['project_name'], version=fname, config=lparams)
    
    trainer = Trainer(logger=wandb_logger, max_epochs=lparams['epochs'], callbacks=callbacks, accelerator="gpu", devices=1)
    trainer.fit(latent_model, latent_loaders[0], latent_loaders[1])    
    
    latent_model.eval()
    def compute2(model, function):
        latent_model.predict_step = function
        outs = tester.predict(latent_model, dataloaders=latent_loaders)
        return [torch.concatenate(vs, dim=0) for vs in outs]
    
    py_logger.info('computing neighbors of centroid...')
    latent2 = compute2(latent_model, latent_model.latent_vectors)
    centroids = [l2.mean(dim=0,keepdim=True) for l2 in latent2] # assume vectorspace here...
    ranges = [torch.maximum(torch.amax(l2-c, dim=0, keepdim=True), torch.amax(c-l2, dim=0, keepdim=True)) for l2,c in zip(latent2,centroids)]
    print(centroids, ranges)
    print(centroids[0].shape, latent2[0].shape)
    
    model.cuda()
    latent_model.cuda()
    
    py_logger.info('computing neighbor scan...')
    def scan(c, r, shp, k=2):
        all_vs = []
        for v in range(c.shape[1]):
            vs = []
            for i in range(-k, k+1):
                ci = c.detach().clone()
                shift = i # * (r[:,v] / k)
                ci[:,v] = ci[:,v] + shift 
                lv = latent_model.decompressor(ci.cuda()).cuda()
                # print('lv', lv.shape)
                # print('ci', ci.shape)
                print('shift', shift / ci[:,v])
                im = model.decompressor(lv, (1, *shp[1:])) # B, C, H, W
                vs.append(im[0].detach().cpu()) # C, H, W
                
                del lv, im, ci
                torch.cuda.empty_cache()
                
            all_vs.append(torch.stack(vs, dim=0))
            # os.system('nvidia-smi')
        return torch.stack(all_vs, dim=0) # v, k, C, H, W
    
    scans = [scan(c,r,shp) for c,r,shp in zip(centroids, ranges, shapes)]
    save('scans', scans) 
    
    from runners.scan_plot import scan_plot
    scan_plot(path=save_path, name='latent_space', labels=labels, data=scans)
    
    del scans
    torch.cuda.empty_cache()
    
    ### switch to resolution scans
    n_levels = 7
    py_logger.info('computing resolution scan data...')
    
    def down(x):
        # downsample and upsample 
        dsize = (x.shape[-2] // 2, x.shape[-1] // 2)
        return torch.nn.functional.interpolate(x, size=dsize, mode='bilinear', align_corners=False)
    
    def up(x, size): 
        return torch.nn.functional.interpolate(x, size=size[-2:], mode='bilinear', align_corners=False)
    
    res_datasets = []
    for d in datasets:
        d0 = d
        ds = [d0]
        for j in range(1,n_levels):
            d0 = down(d0)
            upscaled = up(d0, d.shape)
            assert upscaled.shape == d.shape, f'{upscaled.shape} != {d.shape}'
            ds.append(upscaled)
        res_a = torch.stack(ds, dim=1) # L, B, C, H, W
        res_a = rearrange(res_a, 'b l c h w -> (b l) c h w')
        res_datasets.append(res_a) # fine to coarse
    
    py_logger.info('computing (resolution) latent vectors...')
    loaders3 = [DataLoader(data, shuffle=shuffle, pin_memory=True, **l_kwargs) for data, shuffle in zip(res_datasets, [True, False, False])]
    def res_latent():
        model.predict_step = model.latent_vectors
        outs = tester.predict(model, dataloaders=loaders3)
        outs = [torch.concatenate(vs, dim=0) for vs in outs]
        # now put through latent model
        outs2 = [latent_model.latent_vectors(o.cuda(), 0, 0) for o in outs]
        return [rearrange(vs, '(b l) u -> b l u', l=n_levels).detach().cpu() for vs in outs2]
    res_latents = res_latent()
    save('res_latents', res_latents)
    
    from runners.seq_plot import seq_plot
    [seq_plot(path=save_path, label=labels[i], res_lat=res_latents[i]) for i in range(3)]
    
    # # cat step, dangerous
    # valid = np.concatenate(valid, axis=0)
    # infer = np.concatenate(infer, axis=0)
    
    
    
    # # basic plot
    # from runners.basic_plot import mp4
    # mp4(os.path.join(save_path, 'valid_lviz.mp4'), valid)
    # mp4(os.path.join(save_path, 'infer_lviz.mp4'), infer)