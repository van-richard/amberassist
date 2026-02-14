#!/bin/bash
#SBATCH -p debug
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J reprocess
#SBATCH --array=1

date

# # For Pete  
# module load amber/23-panxl
# module load qmhub
# SANDER="mpirun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"

# For Oscer
source /home/van/oscer.qmmm.env
SANDER="mpiexec.hydra -bootstrap slurm -n 1 -ppn 16 sander.MPI"

export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

init="step3_pbcsetup"
pstep="step5.00_equilibration"
istep="step7_reprocessing"
trajin="step6combined"

window=$(sed -n "$SLURM_ARRAY_TASK_ID"p ../list)
echo "WINDOW ${window}"

cd ../${window}

ln -sf ../reprocess/dedup.sh .
ln -sf ../reprocess/qmhub2.ini .

savepath="$(realpath $(pwd) | realpath $(pwd) |sed 's/\//\\\//g')"
sed "s/__WIN__/${savepath}/" ../reprocess/step7_reprocessing.mdin > step7_reprocessing.mdin
#sed "s/__WIN__/${USER}\/$(basename $(pwd))\/${w}/" ../reprocess/step7_reprocessing.mdin > step7_reprocessing.mdin

$SANDER -O -i ${istep}.mdin -o ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -y ${trajin}.nc -x ${istep}.nc

# Remove duplicates
cd qmhub
bash dedup.sh

date

