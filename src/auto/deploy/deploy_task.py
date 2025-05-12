import logging, os
# from deploy.util import pretty #TODO
logger = logging.getLogger('task')

run_info_file = 'system_parameters.py'

def deploy_task(task_info, task_id, system):
    
    task_info['task_path'] = os.path.join(task_info['save_path'], f"{task_id}-{task_info['task_name']}")
    os.makedirs(task_info['task_path'], exist_ok=True)
    with open(os.path.join(task_info['task_path'],'readme.md'), 'w') as f:
        f.write(task_info.get('description', ''))

    if task_info['task_parameters'] == {}:
        deploy_run(task_info, task_id, 0, system)
    else:
        raise ValueError('Scanning over task parameters is not yet implemented.')
        # TODO define scans
        system['task_parameters'] = {}
        for i, run_parameter_overrides in scans:
            run_info = task_info.copy()
            run_info['run_parameters'].update(run_parameter_overrides)
            deploy_run(run_info, task_id, i, system)
            


def deploy_run(run_info, task_id, run_id, system):
    # now start each run
    run = run_info['task_type']
    run_info['run_id'] = run_id
    runner = os.path.join(run_info['code_path'],'__init__.py')
    tr_id = f'{task_id}_{run_id}'
    param_file = os.path.join(run_info['code_path'], 'parameters', f'{tr_id}.py')
    param = f'parameters.{tr_id}'

    task_path = run_info['task_path']
    run_info['save_path'] = os.path.join(task_path, str(run_id))
    os.makedirs(run_info['save_path'], exist_ok=True)

    sg_engine_script = os.path.join(run_info['scripts_path'], f'sg_engine_{task_id}_{run_id}.sh')
    
    with open(run_info['save_path'] + '/' + run_info_file, 'w') as f:
        f.write(f'system={str(system)}')

    with open(param_file, 'w') as f:
        f.write(f'config={str(run_info)}')

    from templates.templates import sg_engine, Bash, single_run

    os.environ['save_path'] = run_info['save_path']

    logfile = f'{run_info["save_path"]}/.log'
    with Bash(sg_engine_script, sg_engine(**run_info, run_commands=single_run(run, runner, param, logfile.replace('.log', '_python.log')), logfile=logfile.replace('.log', '_engine.log'))):
        if run_info['no_compute'] or (run_info['hostname'] != run_info['cluster_name']):
            logger.info('Starting local task...')
            os.system(f'bash {sg_engine_script}')
        else:
            logger.info('Starting cluster task...')
            
            os.environ["WANDB_MODE"] = "offline" # disable wandb sync  

            os.system(f'qsub {sg_engine_script}')
            logger.info('Starting services...')
            os.system(f"tmux kill-session -t {run_info['project_name']}-services")
            os.system(f"tmux new -A -d -s {run_info['project_name']}-services 'python3 src/auto/services/service_tmux.py; $SHELL'")  
        
        logger.info('Task submitted.')
            