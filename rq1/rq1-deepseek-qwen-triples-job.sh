#!/bin/bash -l
## This file is called `rq1-deepseek-triples-job.sh`
#SBATCH --time=01:00:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1

module load Python

python -c 'import sys; print(sys.version)'

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

python rq1-deepseek-qwen-triples.py /project/home/p200769/data/combined_dataset.sqlite /project/home/p200769/meluxina/rq2/results/rq2_meta-llama_Llama-3.3-70B-Instruct.json
python rq1-deepseek-qwen-triples.py /project/home/p200769/data/combined_dataset.sqlite /project/home/p200769/meluxina/rq2/results/rq2_refined_meta-llama_Llama-3.3-70B-Instruct_generated_by_condensed_llama_ontology.json /project/home/p200769/meluxina/rq2/results/rq2_meta-llama_Llama-3.3-70B-Instruct.json
