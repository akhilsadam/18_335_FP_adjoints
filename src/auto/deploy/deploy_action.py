import repo.git_utils as gl
import logging, os

from deploy.deploy_task import deploy_task
from deploy.util import update, select

code_folder = '.code'
run_folder = 'run'

def deploy_action(action_info, system):
    ## logging
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger('auto-deploy-action')
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logger = rootLogger

    repo = gl.repo(strict=action_info['strict_version_checks'])
    version_data, source_modified_flag = gl.tag_and_version(repo, tag=action_info['strict_version_checks'])
    version, action, save_path = gl.get_save_path(version_data, action_info['strict_version_checks'], action_info['action_name'], action_info['base_save_path'])
    gl.save_env(version, action, save_path)

    fileHandler = logging.FileHandler(f"{save_path}/version.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
        
    hostname = os.uname()[1]
    logger.info(f'hostname: {hostname}')
    code_path = os.path.join(save_path,code_folder)
    action_info.update({
            'save_path': save_path,
            'code_path': code_path,
            'scripts_path': os.path.join(code_path, run_folder),
            'source_modified_flag': source_modified_flag,
            'version_data': version_data,
            'hostname': hostname
        })
    os.makedirs(action_info['code_path'], exist_ok=True)
    os.makedirs(action_info['scripts_path'], exist_ok=True)
    
    # copy all code to task folder
    os.system(f'cp -r src/ml/* {action_info["code_path"]}')
    
    for i,task in enumerate(action_info['tasks']):
        task_info = task.copy()
        update(task_info, action_info, ['tasks'])
        deploy_task(task_info, i, select(system, i=i, k=['actions','tasks']))
