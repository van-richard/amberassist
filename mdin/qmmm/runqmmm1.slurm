#!/bin/bash
#SBATCH -p batch
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %jrunqmmm1.out
#SBATCH -e %jrunqmmm1.err

date

#source /home/van/pete.qmmm.env
module load amber/23-panxl
module load qmhub
module load qchem/5.2

export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"
#source /home/van/oscer.qmmm.env

init="step3_pbcsetup"
istep="step5.00_equilibration"
#n_windows=$(cat list | wc -l)

#for i in $(seq 37 41); do
for i in $(echo 41); do
    printf -v window "%02d" $i
    
    cd $window
    $SANDER -O -i ${istep}.mdin -o ${istep}.mdout -p ${init}.parm7 -c ${istep}_inp.ncrst -r ${istep}.ncrst -x ${istep}.nc -inf ${istep}.mdinfo

    printf -v j "%02d" $(($i + 1))
    cp sinrvels.rst ../${j}/

    cd ..
done

date

