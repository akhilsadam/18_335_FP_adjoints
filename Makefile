.ONESHELL:
SHELL=/bin/bash

# allowed input arguments
# ngpu \in Z^+

# folder structure
auto=src/auto
init=${auto}/init
install=${auto}/install
stable_deploy_path='../stable/'
project_name='delay-neural-operator'
REPO='git@github.com:akhilsadam/delay-neural-operator.git' # TODO change to MSEAS repo...
key_location='/home/${USER}/.ssh/id_ed25519_g'
# change above two to your repo if you are a new administrator with main push permissions
JPORT=8596

# make commands
system_run: # env configuration on login nodes first for automation (and since cluster is air-gapped)
	chmod +x ${init}/conda_up.sh
	chmod +x ${init}/venv_up.sh
	source ${init}/conda_up.sh
	source ${init}/venv_up.sh
	uv pip install -r ${install}/requirements.txt
	python3 ${auto}/deploy_system.py
	- deactivate
	- conda deactivate

run: # env reinstall serves as audit (since cluster is air-gapped)
	source ${init}/conda_up.sh
	source ${init}/venv_up.sh
	uv pip install -r ${install}/requirements.txt
	python3 src/train.py
	- deactivate
	- conda deactivate

_ins:
	source ${init}/conda_up.sh
	source ${init}/venv_up.sh
	uv pip install -r ${install}/requirements.txt
	- deactivate
	- conda deactivate

update_source:
	source ${init}/conda_up.sh
	eval `ssh-agent` && ssh-add ${key_location}
	# Now we will update the source code
	git status
	echo "DO NOT PROCEED IF there are uncommitted changes; make sure the last commit is test-worthy."
	read -p "Please confirm [enter] to continue with the push to the master branch:"
	git pull
	git push
	git push origin ${USER}-auto-dev:main
	read -p "Please confirm [enter] to continue with the pull to the stable folder:"
	mkdir -p ${stable_deploy_path}
	cd ${stable_deploy_path}
	[ -d ${project_name} ] && cd ${project_name} && git pull || git clone ${REPO}
	- conda deactivate

local_web:
	source ${init}/conda_up.sh
	source ${init}/venv_up.sh
	uv pip install -r ${install}/web_requirements.txt
	cd ${local_web}
	jupyter lite build
	- deactivate
	- conda deactivate

jupyter: # no password/token needed since this will already be behind auth.
	source ${init}/conda_up.sh
	source ${init}/venv_up.sh
	uv pip install -r ${install}/web_requirements.txt
	jupyter lab --IdentityProvider.token='' --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy":"self http://mseas.mit.edu:${JPORT}/ "}}' --ServerApp.ip='0.0.0.0' --ServerApp.allow_origin="https://mseas.mit.edu" --ServerApp.disable_check_xsrf=True --port ${JPORT} --no-browser
	- deactivate
	- conda deactivate