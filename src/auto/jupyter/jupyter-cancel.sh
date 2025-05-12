#!/bin/bash
su - $1 <<!
$2
echo $whoami
qdel scale_invariants_jupyter
!