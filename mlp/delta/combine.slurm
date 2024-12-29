#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=combine

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate

python combine.py
#python combine1.py
#python combine_sqm.py

date
