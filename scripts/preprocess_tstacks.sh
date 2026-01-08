#!/bin/bash
#SBATCH --job-name=proc_tstacks
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
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312

# install xarray if not available (into user space)
pip install --user xarray netcdf4

# Set paths
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
TSTACK_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks"
AREA_FILE="/oak/stanford/groups/cyaolai/JoshRines/data/all_lakes_2019.nc"
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed"
MAX_LAKES=20  # Set to empty string for all lakes, or a number to limit

# Create output directory
mkdir -p $OUTPUT_DIR

# Run preprocessing
if [ -z "$MAX_LAKES" ]; then
    python3 $REPO_DIR/scripts/preprocess_tstacks.py \
        --tstack_dir $TSTACK_DIR \
        --area_file $AREA_FILE \
        --output_dir $OUTPUT_DIR
else
    python3 $REPO_DIR/scripts/preprocess_tstacks.py \
        --tstack_dir $TSTACK_DIR \
        --area_file $AREA_FILE \
        --output_dir $OUTPUT_DIR \
        --max_lakes $MAX_LAKES
fi

echo "Done!"