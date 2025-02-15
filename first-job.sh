#!/bin/bash -l
## This file is called `MyFirstJob_MeluXina.sh`
#SBATCH --time=00:00:10
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1

echo 'Hello, world!'

nvidia-smi
