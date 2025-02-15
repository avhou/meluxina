#!/bin/bash -l
## This file is called `python-job.sh`
#SBATCH --time=00:05:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1

module load Python

python -c  'import sys; print(sys.version)'

python -m venv venv

source venv/bin/activate

pip --cache-dir=/project/home/p200769/data/pip install -r requirements.txt

python run-models.py
