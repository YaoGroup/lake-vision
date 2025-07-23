#!/bin/bash
#SBATCH --job-name=IsAsLSTM
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/IsAsLSTM/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/IsAsLSTM/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH -p serc
#SBATCH --gpus=1 --constraint GPU_MEM:80GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
python3 /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/ConvLSTM_areaseq.py


# --constraint GPU_SKU:A100_SXM4