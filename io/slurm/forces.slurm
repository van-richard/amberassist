#!/bin/bash
#SBATCH --partition=normal
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=1G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=02:00:00
#SBATCH --job-name=fnpy
#SBATCH --array=1-22

date

module load intel/2020a
source /home/panxl/qchem/trunk2/setqc.sh
source /home/panxl/amber20/amber.sh

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate

w=$(sed -n "$SLURM_ARRAY_TASK_ID"P list)

cd ../${w}
cp ../ml/dft_forces.py .
sed -i "s/__WIN__/${w}/" dft_forces.py
python dft_forces.py

date
