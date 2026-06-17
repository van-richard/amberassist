#!/bin/bash
#SBATCH -p express
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %jrun_plumed_md.out
#SBATCH -e %jrun_plumed_md.err

date

module load intel/2021.2.0
module load impi/2021.2.0
source /scratch/van/Programs/sander26/amber.sh

export PLUMED_ROOT=/scratch/van/Programs/plumed-2.9.4
export LD_LIBRARY_PATH=/scratch/van/Programs/plumed-2.9.4/lib:${LD_LIBRARY_PATH}
export PATH=/scratch/van/Programs/plumed-2.9.4/bin:${PATH}
export PLUMED_KERNEL=/scratch/van/Programs/plumed-2.9.4/lib/libplumedKernel.so

export MKLROOT=/opt/intel/oneapi/mkl/2021.2.0
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_THREAD_YIELD=3
export I_MPI_THREAD_SLEEP=200

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
#export OMP_DYNAMIC=FALSE
#export MKL_DYNAMIC=FALSE

#export OMP_WAIT_POLICY=PASSIVE
#export KMP_BLOCKTIME=0
#export KMP_SETTINGS=TRUE

idx="0"
printf -v i "%02d" ${idx}
istep="step5.${i}_equilibration_plumed_md"
mdin="step5.00_equilibration.mdin"
pstep="step5.00_equilibration_inp"
init="step3_pbcsetup"

plumed_input="${PLUMED_INPUT:-plumed.dat}"
echo "Requested PLUMED input: ${plumed_input}"

SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"

/usr/bin/time -v \
$SANDER -O -i ${mdin} -o ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -r ${istep}.ncrst -x ${istep}.nc -ref ${init}.ncrst -inf ${istep}.mdinfo

date

echo "${SLURM_JOBID} ${OMP_NUM_THREADS}threads,dynamic${OMP_DYNAMIC},${OMP_WAIT_POLICY},${plumed_input}" >> notes
