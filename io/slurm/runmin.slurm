#!/bin/bash
#SBATCH -p express
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=32
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J min

date

module load amber/23-gpu

init="step3_pbcsetup"

pmemd="mpirun -n ${SLURM_NTASKS} pmemd.MPI"

# Run minimization (generally best on CPU)
$pmemd -i min.mdin -p ${init}.parm7 -c ${init}.rst7 -O -o min.mdout -r min.rst7 -ref ${init}.rst7 

date

