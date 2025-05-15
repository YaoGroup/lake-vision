#!/bin/bash
#SBATCH --job-name=stack_s2
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sherlock/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sherlock/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# module loading:
module purge
module load python/3.12

# pip install --upgrade --force-reinstall -e git+https://github.com/jharlanr/sat-tile-stack.git#egg=sat_tile_stack
# pip freeze > requirements.txt
# pip install -r requirements.txt

# activate virtual environment
source /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/.venv/bin/activate

# run script:
python3 /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/lakevision/s2_stacking.py