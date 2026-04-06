#!/bin/bash
#SBATCH --job-name=lv_sweep
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu
#SBATCH --array=0-5

# =============================================================================
# WANDB SWEEP RUNNER (6 parallel agents)
# =============================================================================
#
# Launches 6 wandb agents as a SLURM array job.
# Each agent picks the next unfinished run from the sweep grid.
# Each agent gets its own GPU, so 6 runs execute simultaneously.
#
# USAGE:
#   wandb sweep sweep_ed_split.yaml    # creates sweep, prints sweep ID
#   sbatch run_sweep.sh <sweep_id>     # launches 6 parallel agents
#
# =============================================================================

# Check sweep ID argument
SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: No sweep ID provided."
    echo "Usage: sbatch run_sweep.sh <sweep_id>"
    exit 1
fi

# Set paths
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"

# Create directories
mkdir -p "$SHERLOCK_DIR/logs"

echo "=============================================="
echo "Lake Vision Sweep Agent"
echo "=============================================="
echo "Sweep ID:   $SWEEP_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Node:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

# Install additional dependencies
pip install --user xarray netcdf4

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"

cd "$REPO_DIR/engine/training"

echo ""
echo "Starting wandb agent..."
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Each agent will pick runs from the grid until all 16 are done.
# With 6 agents and 16 runs, each agent runs ~2-3 experiments.
wandb agent "$SWEEP_ID"

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
echo "Agent $SLURM_ARRAY_TASK_ID finished"
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m (${DURATION_SEC}s total)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="
echo ""
echo "To sync wandb runs, run:"
echo "  cd $SHERLOCK_DIR && wandb sync wandb/offline-run-*"
echo ""

exit $EXIT_CODE
