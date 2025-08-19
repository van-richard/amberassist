#!/bin/bash
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --mem=0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=48:00:00
#SBATCH --job-name=qchem
#SBATCH --array=1-7

set -e
hostname

module load intel/2020a
source /home/panxl/qchem/trunk2/setqc.sh
source /home/panxl/amber20/amber.sh

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate

export UCX_TLS=rc,sm,self
export I_MPI_PIN=0
export I_MPI_THREAD_YIELD=3
export I_MPI_THREAD_SLEEP=200
export I_MPI_OFI_PROVIDER=verbs

# For QMHub only
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

date

w=$(sed -n "$SLURM_ARRAY_TASK_ID"P list)

echo "window ${w}"

cd ../${w}
rm training_set1.py
ln -sf ../ml/training_set1.py .
ln -sf ../ml/qm_elem.txt .
python -u training_set1.py

date

