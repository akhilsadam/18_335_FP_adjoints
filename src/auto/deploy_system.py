import os
from parameters.default_configuration import *

from deploy.deploy_action import deploy_action
from deploy.util import select

from system_parameters import system
system = sys_template.validate(system)

for i, action in enumerate(system['actions']):
    action_config = sys_config.copy()
    action_config.update(default_action_config)
    action_config.update(action_template.validate(action))
    
    deploy_action(action_config, select(system, i, ['actions',]))
    

