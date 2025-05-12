#!/bin/bash --login
source ~/.bashrc
HOSTNAME=$(hostname)
echo $HOSTNAME
if [[ $HOSTNAME == mseas.mit.edu ]] || [[ $HOSTNAME == ibgpu-compute-0-0.local ]]
then {
    if [ -d ".sw" ]
    then echo 'software environment already exists; activating'
    else conda create --prefix .sw python=3.10.15 git tmux graphviz conda-forge::ffmpeg
    fi
    conda activate $(pwd)/.sw && python3 --version
}
else echo 'not on cluster; fallback to pip / local python'
fi