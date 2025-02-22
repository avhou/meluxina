#!/bin/bash -l
## This file is called `translate-split-job.sh`
#SBATCH --time=20:00:00
#SBATCH --account=p200769
#SBATCH --partition=gpu
#SBATCH --qos=default
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

time python translate-split-hits.py cuda:0 Helsinki-NLP/opus-mt-fr-en /project/home/p200769/data/hits-fr-0.csv /project/home/p200769/data/hits-fr-0-translated.csv &
pid1=$!

time python translate-split-hits.py cuda:1 Helsinki-NLP/opus-mt-fr-en /project/home/p200769/data/hits-fr-1.csv /project/home/p200769/data/hits-fr-1-translated.csv &
pid2=$!

time python translate-split-hits.py cuda:2 Helsinki-NLP/opus-mt-fr-en /project/home/p200769/data/hits-fr-2.csv /project/home/p200769/data/hits-fr-2-translated.csv &
pid3=$!

time python translate-split-hits.py cuda:3 Helsinki-NLP/opus-mt-fr-en /project/home/p200769/data/hits-fr-3.csv /project/home/p200769/data/hits-fr-3-translated.csv &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

echo "All FR scripts have finished."

time python translate-split-hits.py cuda:0 Helsinki-NLP/opus-mt-nl-en /project/home/p200769/data/hits-nl-0.csv /project/home/p200769/data/hits-nl-0-translated.csv &
pid5=$!

time python translate-split-hits.py cuda:1 Helsinki-NLP/opus-mt-nl-en /project/home/p200769/data/hits-nl-1.csv /project/home/p200769/data/hits-nl-1-translated.csv &
pid6=$!

time python translate-split-hits.py cuda:2 Helsinki-NLP/opus-mt-nl-en /project/home/p200769/data/hits-nl-2.csv /project/home/p200769/data/hits-nl-2-translated.csv &
pid7=$!

time python translate-split-hits.py cuda:3 Helsinki-NLP/opus-mt-nl-en /project/home/p200769/data/hits-nl-3.csv /project/home/p200769/data/hits-nl-3-translated.csv &
pid8=$!

wait $pid5 $pid6 $pid7 $pid8

echo "All NL scripts have finished."
