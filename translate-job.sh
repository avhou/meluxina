#!/bin/bash -l
## This file is called `translate-job.sh`
#SBATCH --time=00:15:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1

module load Python

python -c  'import sys; print(sys.version)'

python -m venv venv

source venv/bin/activate

python -m pip --cache-dir=/project/home/p200769/data/pip install -r requirements.txt

echo "dependencies installed"

export PROJECT_DATA_DIR=/project/home/p200769/data
export INPUT_FILE=filtered-hits-all.sqlite
export OUTPUT_FILE=translated-filtered-hits-all.sqlite
export HF_HOME=/project/home/p200769/data/huggingface
export HUGGINGFACE_HUB_CACHE=/project/home/p200769/data/huggingface

echo "starting the python script"

python translate-hits.py
