#!/bin/bash
#SBATCH --job-name=xv_attn
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/xv_attn/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/xv_attn/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH -p serc
#SBATCH --gpus=1 --constraint GPU_MEM:80GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

## =========== ##
# module loading:
## =========== ##
ml system
ml python/3.12.1

## ========== ##
# activate venv:
## ========== ##
source ~/.venvs/lakesenv/bin/activate

## ======= ##
# run script:
## ======= ##
python3 /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/clstmxv_attn.py


# --constraint GPU_SKU:A100_SXM4