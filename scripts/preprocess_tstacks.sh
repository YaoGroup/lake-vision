#!/bin/bash
#SBATCH --job-name=proc_tstax
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# module loading:
module purge
module load python/3.12

# activate virtual environment
source /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/.venv/bin/activate

# Set paths - EDIT THESE
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
TSTACK_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks"  # EDIT: path to your raw tstacks
AREA_FILE="/oak/stanford/groups/cyaolai/JoshRines/data/all_lakes_2019.nc"  # EDIT: path to area sequences
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/processed"
MAX_LAKES=50  # Set to empty string for all lakes: MAX_LAKES=""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run preprocessing
python3 $REPO_DIR/scripts/preprocess_lakes.py \
    --tstack_dir $TSTACK_DIR \
    --area_file $AREA_FILE \
    --output_dir $OUTPUT_DIR \
    --max_lakes $MAX_LAKES

echo "Done!"