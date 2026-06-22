#!/bin/bash
#SBATCH -p guest-compute
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %jrun_plumed_qmmm_metad.out
#SBATCH -e %jrun_plumed_qmmm_metad.err

date

module load sander/26
module load plumed/2.9.4
module load qchem/5.2
module load qmhub2

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

idx="0"
printf -v i "%02d" ${idx}
istep="step5.${i}_equilibration_plumed_qmmm_metad"
mdin="step5.00_equilibration.mdin"
pstep="step5.00_equilibration_inp"
init="step3_pbcsetup"

# Select the PLUMED input to test after verifying the local Sander PLUMED hook.
plumed_input="${PLUMED_INPUT:-plumed.dat}"
echo "Requested PLUMED input: ${plumed_input}"

#SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"
SANDER="mpirun --bind-to none --map-by slot -np ${SLURM_NTASKS_PER_NODE} sander.MPI"

# /usr/bin/time -v \
$SANDER -O -i ${mdin} -o ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -r ${istep}.ncrst -x ${istep}.nc -ref ${init}.ncrst -inf ${istep}.mdinfo

date

#echo "${SLURM_JOBID} ${OMP_NUM_THREADS}threads,dynamic${OMP_DYNAMIC},${OMP_WAIT_POLICY},${plumed_input}" >> notes
