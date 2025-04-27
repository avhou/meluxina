#!/bin/bash -l
## This file is called `rq2-deepseek-job.sh`
#SBATCH --time=01:00:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1

module load Python

python -c  'import sys; print(sys.version)'

python -m venv venv

source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip --cache-dir=/project/home/p200769/data/pip install -r requirements.txt

export PROJECT_DATA_DIR=/project/home/p200769/data
export HF_HOME=/project/home/p200769/data/huggingface
export HUGGINGFACE_HUB_CACHE=/project/home/p200769/data/huggingface
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=/project/home/p200769/data/cuda_cache
export CUDA_CACHE_MAXSIZE=4294967296
export HUGGINGFACEHUB_API_TOKEN=$TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python rq2-llama.py --prompts /project/home/p200769/data/prompts-rebel.json --group-by chunk
