#!/bin/bash
su - $1 <<!
$2
echo $whoami
cd $PWD/../../../
qsub -terse src/auto/jupyter/sg_jupyter.sh 
!