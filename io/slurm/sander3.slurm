#!/bin/bash
#SBATCH --partition=32gb_20core
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=07:00:00
#SBATCH --job-name=sander

JOBS=`seq 0 0`
# JOBS=$1
DIRS=`seq -f"%02g" 0 59`

for i in $JOBS; do
    parallel --workdir ${SLURM_SUBMIT_DIR} --progress -P ${SLURM_NTASKS} ./run_sander.sh ::: $DIRS ::: $i
done
