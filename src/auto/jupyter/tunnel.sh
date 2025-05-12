#!/bin/bash
su - $1 <<!
$2
cd /home/$1/.ssh/
echo $pwd
ls
eval `ssh-agent`
ssh-add id_ed25519
ssh -fNL 127.0.0.1:8596:127.0.0.1:8596 ibgpu-compute-0-0.local
!