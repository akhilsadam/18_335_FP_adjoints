from schema import Schema, And, Use, Optional, Or

sys_config = {
    'project_name': 'delay-neural-operator',
    'cluster_name': 'mseas.mit.edu',
    'base_save_path': 'run/',
}

default_action_config = {
    'action_name': 'test',
    'no_compute': False, # for simple testing work
    'strict_version_checks': False,
    'save_frequency': 100,
    'debug': False,
    'ngpu': 1,
    'data': "data/MassBay/",
}

run_template = Schema({
    Or('model', 'analysis'): str,
    Optional('model_parameters'): dict,
    Optional('loader_parameters'): dict,
    Optional('load_from'): str,
    Optional('test_only'): bool,
})

task_template = Schema({
    'task_name': str,
    'task_parameters': dict,
    'task_type': str,
    'run_parameters': run_template,
    Optional('task_parameters'): dict,
    }
)

action_template = Schema(
    {
        'action_name': str,
        'no_compute': bool,
        'strict_version_checks': bool,
        'save_frequency': int,
        'ngpu': int,
        'data': str,
        'tasks': [task_template,],
        # other info...
        Optional(object): object
    }
)

sys_template = Schema(
    {
        'actions': [action_template,],
    }
)