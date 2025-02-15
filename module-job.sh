#!/bin/bash -l
## This file is called `module-job.sh`
#SBATCH --time=00:05:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1

module avail 

module list 

