#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1

date

module load amber/23-gpu

cpu="mpirun -n ${SLURM_NTASKS_PER_NODE} pmemd.MPI"
pmemd="pmemd.cuda"
parm="step3_pbcsetup"

if [ -f "${parm}_1264.parm7" ]; then
    init="${parm}_1264"
else
    init="${parm}"
fi

# min
#$cpu -i min.mdin -p ${init}.parm7 -c ${init}.rst7 -O -o min.mdout -r min.rst7 -inf min.mdinfo -ref ${init}.rst7 -x min.nc

# Heating
#$cpu -i heat.mdin -p ${init}.parm7 -c min.rst7 -O -o heat.mdout -r heat.rst7 -inf heat.mdinfo -ref ${init}.rst7 -x heat.nc

# Pressure
$pmemd -i density.mdin -p ${init}.parm7 -c heat.rst7 -O -o density.mdout -r density.rst7 -inf density.mdinfo -ref ${init}.rst7 -x density.nc

# NVT
$pmemd -i prod.mdin -p ${init}.parm7 -c density.rst7 -O -o prod00.mdout -r prod00.ncrst -inf prod00.mdinfo -ref ${init}.rst7 -x prod00.nc

date

