# automation 

initialization [`init`]
- `sg_up` starts job (if on cluster)
- `conda_up` activates conda (if on cluster)
- `venv_up` activates virtualenv
- runs python

repository management [`repo`]
- automates git tagging and versioning
- creates Singularity def file & images

scheduler [`sched`]
- task automation
- handles job submission and matrix jobs

installation requirements [`install`]
- `requirements.txt`

Complete feature installs require one set of the following:
- `>=Python3.10, git, singularity, tmux`
- recent `conda`

Alternatively, generated images can be used with just `singularity`.