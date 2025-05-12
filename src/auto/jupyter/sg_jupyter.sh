#!/bin/bash
#
#$ -cwd
#$ -V
#$ -j y
#$ -w e
#$ -S /bin/bash
#$ -l h_core=0
#### -l gpu=1 #### no need for gpu
#$ -N scale_invariants_jupyter
#$ -q ibgpu.q
#$ -o jup.log

date
make -B jupyter
date