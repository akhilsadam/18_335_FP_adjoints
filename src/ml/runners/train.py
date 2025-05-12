import os
import importlib
import logging

import torch
torch.set_float32_matmul_precision('high') # can do medium as well

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import numpy as np

seed_everything(1)
from iox.load import loaders as _loaders
from utilities.repo import understand_env
# from utilities.gpulog import GPU_util # TODO

# torch.autograd.set_detect_anomaly(True)


def run(config, py_logger):
    run_p = config['run_parameters']

    version, task, save_path = understand_env()
    fname = '-'.join(config['task_path'].split('/')[1:]) + f"-{config['run_id']}"
    
    py_logger.info(f'Logging to wandb with {fname}')
    
    # which gpus are used?
    cvs = os.environ['CUDA_VISIBLE_DEVICES']
    cvs = [int(c) for c in cvs.split(',')] # convert to list of ints
    config.update({'which_gpu': cvs})
    
    # os.makedirs(os.path.join(save_path,"wandb/"), exist_ok=True)
    wandb_logger = WandbLogger(project=config['project_name'], name=fname,
                    version=fname, config=config)
                    # save_dir=save_path) # has issues with sync...
    py_logger.info(f'wandb configured')
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_path,"ckpt/"),
            every_n_train_steps=config['save_frequency'],
        ),
        # GPU_util(wandb_logger) # TODO
    ]
    
    model = importlib.import_module(config['run_parameters']['model'])
        
    if run_p.get('load_from', False):
        model = model.Network.load_from_checkpoint(run_p['load_from'])
        
        loaders, shapes, _ = _loaders(config['data'], batch_size=model.param.batch_size,
                               **config['run_parameters'].get('loader_parameters', {}))
        
        py_logger.info(f'loaded model from {run_p["load_from"]}')
        
    else:
        parameters = model.Parameters().update(**config['run_parameters']['model_parameters'])
        config['run_parameters']['model_parameters'].update(parameters.__dict__) # so now they are the same
        
        
        loaders, shapes, _ = _loaders(config['data'], batch_size=parameters.batch_size,
                               **config['run_parameters'].get('loader_parameters', {}))
        
        parameters.set_shapes(shapes)
        model = model.Network(parameters)
        py_logger.info('model initialized')
    
    if not run_p.get('test_only', False):
        py_logger.info(f'Rank: {rank_zero_only.rank}')
        if rank_zero_only.rank == 0: # workaround for multi-gpu wandb logging        
            wandb_logger.watch(model, log='all', log_freq=config['save_frequency'])
            py_logger.info('wandb: enabled model watch')
              
        trainer = Trainer(
            max_steps=parameters.it, accelerator="gpu", devices=config['ngpu'], logger=wandb_logger, strategy="ddp", callbacks=callbacks
        )    
        py_logger.info('trainer initialized')
        trainer.fit(model, loaders[0], loaders[1])
        py_logger.info('training finished')
    else:
        py_logger.info('skipping training...')

    model.eval() # may be redundant
    py_logger.info('testing...')
    tester = Trainer(
        accelerator="gpu", devices=1, logger=wandb_logger
    )    
    valid, infer = tester.predict(model, dataloaders=(loaders[1], loaders[2]))
    py_logger.info('tester finished')
    
    # cat step, dangerous
    valid = np.concatenate(valid, axis=0)
    infer = np.concatenate(infer, axis=0)
    
    np.savez(os.path.join(save_path, 'valid'), valid)
    np.savez(os.path.join(save_path, 'infer'), infer)
    
    # basic plot
    from runners.basic_plot import mp4
    mp4(os.path.join(save_path, 'valid.mp4'), valid)
    mp4(os.path.join(save_path, 'infer.mp4'), infer)