#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH -J md

date

init="step3_pbcsetup"
pmemd="pmemd.cuda"
module="amber/23-gpu"

module load ${module}

# Heating
$pmemd -i heat1.mdin -p ${init}.parm7 -c min.rst7 -O -o heat1.mdout -r heat1.rst7 -inf heat1.mdinfo -ref ${init}.rst7 -x heat1.nc
$pmemd -i heat2.mdin -p ${init}.parm7 -c heat1.rst7 -O -o heat2.mdout -r heat2.rst7 -inf heat2.mdinfo -ref ${init}.rst7 -x heat2.nc

# Pressure
$pmemd -i density.mdin -p ${init}.parm7 -c heat2.rst7 -O -o density.mdout -r density.rst7 -inf density.mdinfo -ref ${init}.rst7 -x density.nc

# NVT
$pmemd -i prod.mdin -p ${init}.parm7 -c density.rst7 -O -o prod00.mdout -r prod00.rst7 -inf prod00.mdinfo -ref ${init}.rst7 -x prod00.nc

date


