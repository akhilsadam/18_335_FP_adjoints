import sys, os
import importlib
import logging

from utilities.set_gpu import set_gpu

args = sys.argv[1:]
# action, parameters, logfile

## logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger('auto')
rootLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

fileHandler = logging.FileHandler(args[-1])
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

def run():
    # slightly unsafe python code execution
    try:
        rootLogger.info(f'Running auto with {args[0]} and {args[1]}')
        rootLogger.info(sys.argv)
        work = importlib.import_module(args[0])
        config = importlib.import_module(args[1]).config
        
        set_gpu(int(config['ngpu']))
        os.system('echo $CUDA_VISIBLE_DEVICES')
        
        if config['debug']:
            rootLogger.setLevel(logging.DEBUG)
            from debug.anzen_torch import AnzenTorchTracer, AnzenSuite
            att = AnzenTorchTracer(
                ['__init__.py','train.py'], # files to check inside
                [],# [AnzenSuite.maxmin_forward, AnzenSuite.nan_forward,],   # forward checks
                [],# [AnzenSuite.maxmin_gradient, AnzenSuite.nan_gradient,], # backward checks
                fwd_timeout=0 # timeout for forward pass in seconds
                )            
            # os.environ['NCCL_DEBUG'] = 'INFO'
            # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
            
        rootLogger.info('Starting work')
        work.run(config = config, py_logger=rootLogger)
        rootLogger.info('Finished work')
    except Exception as e:
        rootLogger.error(f'Error in auto: {e}')
        raise e
    
if __name__ == '__main__':
    run()