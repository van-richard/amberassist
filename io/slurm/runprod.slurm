#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 3-12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1

date

module load amber/23-gpu

init="step3_pbcsetup_1264"
pmemd="pmemd.cuda"


if [ -z $1 ] || [ -z $2 ]; then
    echo "Missing: $pstep and/or $istep values"
    exit 1
fi
printf -v p "%02d" $1
printf -v i "%02d" $2


pstep="prod${p}"
istep="prod${i}"
echo "$pstep $istep"

$pmemd -O -i prod2.mdin \
    -p ${init}.parm7 \
    -c ${pstep}.rst7 \
    -o ${istep}.mdout \
    -r ${istep}.rst7 \
    -inf ${istep}.mdinfo \
    -ref ${init}.rst7 \
    -x ${istep}.nc

date
