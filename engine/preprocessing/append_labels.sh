#!/bin/bash
#SBATCH --job-name=append_labels
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# module loading:
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

# install xarray if not available (into user space)
pip install --user xarray netcdf4

# Set paths
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
INPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_with_cloudyseq"
OUTPUT_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_labeled"
LABELS_FILE="/oak/stanford/groups/cyaolai/JoshRines/data/labels_2019_volumes_CW.csv"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "Reading .nc files from: $INPUT_DIR"
echo "Appending labels from:  $LABELS_FILE"
echo "Writing labeled files to: $OUTPUT_DIR"
echo ""

# Run in labels-only mode
python3 $REPO_DIR/engine/preprocessing/preprocess_tstacks.py \
    --labels_only \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --labels_file $LABELS_FILE

echo "Done!"