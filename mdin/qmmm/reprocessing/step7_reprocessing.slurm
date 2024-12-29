#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=reprocess
#SBATCH --array=16-22

date
module load amber/23-panxl
module load qchem/5.2
module load qmhub

export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"

init="step3_pbcsetup"
pstep="step5.01_equilibration_inp"
istep="step7_reprocessing"
traj="all_step6"
#traj="step7_reprocessing"
basedir=$(basename $(pwd))
model=$(basename $(realpath ../))

# Print this working directory and window
window=$(sed -n "$SLURM_ARRAY_TASK_ID"p ../list)
echo "Window ${window}"

qmhubpath="/tmp/${USER}/${model}/${window}" 
mkdir -p ${qmhubpath}
cd ../${window}
ln -sf ../${basedir}/dedup.sh .
ln -sf ../${basedir}/qmhub2.ini .

sed "s/__WIN__/${model}\/${window}/" ../${basedir}/${istep}.mdin > ${istep}.mdin

$SANDER -O -i ${istep}.mdin -p ${init}.parm7 -c ${pstep}.ncrst -o ${istep}.mdout -y ${traj}.nc -x ${istep}.nc -ref ${pstep}.ncrst

date

cd $qmhubpath
mksquashfs  qmhub qmhub.squashfs
cd -
mv ${qmhubpath}/qmhub.squashfs .
unsquashfs -d qmhub qmhub.squashfs 

# Clean up...
cd qmhub
bash ../dedup.sh

date

