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

export PROJECT_DATA_DIR=/project/home/p200769/data
export INPUT_FILE=filtered-hits-all.sqlite
export OUTPUT_FILE=translated-filtered-hits-all.sqlite
export HF_HOME=/project/home/p200769/data/huggingface
export HUGGINGFACE_HUB_CACHE=/project/home/p200769/data/huggingface
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/project/home/p200769/data/cuda_cache
export CUDA_CACHE_MAXSIZE=4294967296

python translate-split-hits.py cuda:0 /project/home/p200769/data/hits-fr-0.csv &
pid1=$!

python translate-split-hits.py cuda:1 /project/home/p200769/data/hits-fr-1.csv &
pid2=$!

python translate-split-hits.py cuda:2 /project/home/p200769/data/hits-fr-2.csv &
pid3=$!

python translate-split-hits.py cuda:3 /project/home/p200769/data/hits-fr-3.csv &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

echo "All scripts have finished."
