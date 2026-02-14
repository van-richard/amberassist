#!/bin/bash
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J reprocess
#SBATCH --array=1

date

module load amber/23-panxl
module load qmhub

export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"

init="step3_pbcsetup"
pstep="step5.00_equilibration"
istep="step7_reprocess"
trajin="step6"

window=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ../list)
echo "WINDOW ${window}"

cd ../${window}

ln -sf ../reprocess/qmhub2.ini .
bash ../reprocess/write_mdin.sh "../reprocess" "step7_reprocess.mdin" "${window}"

$SANDER -O -i ${istep}.mdin -o  ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -y ${trajin}*.nc -x ${istep}.nc

# if it works, run this to remove duplicates `./dedup.sh`
ln -sf ../reprocess/dedup.sh .
# bash dedup.sh

# compress qmhub2
# module load base
# mksquashfs qmhub2 qmhub2.squashfs

# clean up
rm qmhub2.ini

date

