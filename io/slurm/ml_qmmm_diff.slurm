#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --mem=0G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=48:00:00
#SBATCH --job-name=ml

set -e
hostname

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate
#eval "$(/home/van/ourdisk/.Programs/miniconda3/bin/conda shell.bash hook)"
#conda activate pytorch_cuda

python -u ml_qmmm_diff.py

date
