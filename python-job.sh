#!/bin/bash -l
## This file is called `MyFirstJob_MeluXina.sh`
#SBATCH --time=00:00:10
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1

module load Python

python -c  'import sys; print(sys.version)'

