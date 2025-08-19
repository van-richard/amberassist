#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=sqm

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

#w=$(sed -n "$SLURM_ARRAY_TASK_ID"P list)
#echo "window ${w}"
#cd ../${w}
#rm sqm_training_set.py || true
#rm -r tmp || true 
#cp ../ml/sqm_training_set.py .
#python sqm_training_set.py
#mkdir -p tmp
#cp -rL /dev/shm/run_* tmp/

seq -w 0 41 | while read i; do
echo "window $i"
cd ../$i
ln -sf ../ml/sqm_training_set.py .
ln -sf ../ml/qm_elem.txt .
python -u sqm_training_set.py 
done


date

