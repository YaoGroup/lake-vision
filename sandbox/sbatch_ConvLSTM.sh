#!/bin/bash
#SBATCH --job-name=ConvLSTM
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/ConvLSTM/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/logs/ConvLSTM/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH -p serc
#SBATCH --gpus=1 --constraint GPU_SKU:A100_SXM4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# #################
# # clean Python env:
# #################
# export PYTHONNOUSERSITE=True 

#################
# module loading:
#################
ml system # good
ml python/3.12.1 # good
ml py-numpy/1.26.3_py312 # good
ml py-pandas/2.2.1_py312 # good
ml viz # good, needed for matplotlib
ml py-matplotlib/3.8.3_py312 # good
ml py-scipy/1.12.0_py312 # good
ml py-h5py/3.10.0_py312 # good
ml py-pytorch/2.2.1_py312 # good, loads cuda automatically, reloads gcc/14.2.0 => gcc/10.1.0 (not sure if that's bad or not)
ml py-torchvision/0.17.1_py312 # good
ml py-scikit-learn/1.5.1_py312 # good


########################################
## ENSURE netCDF4 IS AVAILABLE LOCALLY ##
########################################
# Set up local user install path
export PYTHONUSERBASE=$HOME/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
# Install netCDF4 only if not already present
if ! python3 -c "import netCDF4" &> /dev/null; then
    echo "Installing netCDF4 for xarray..."
    pip3 install --user netCDF4
fi
if ! python3 -c "import h5netcdf" &> /dev/null; then
    echo "Installing h5netcdf for xarray..."
    pip3 install --user h5netcdf
fi

#############
# run script:
#############
python3 /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/ConvLSTM.py

