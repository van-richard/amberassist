#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -o %jrun.out
#SBATCH -e %jrun.err

date

module load amber/23-panxl
module load qchem/5.2
module load qmhub

# vbatch run.slurm $WINDOW 
if [ -z "${1}" ]; then
    echo "missing: \$WINDOW"
    echo "try: vbatch run.slurm \$WINDOW"
    exit 1
else
    WINDOW=$(printf %02d ${1})
fi

## run step6.${idx}
idx=${IDX:-6}
nstlim=${NSTLIM:-800}
printf -v i "%02d" ${idx}
istep="step6.${i}_equilibration"

if [ "${idx}" -eq "0" ]; then
    pstep="step5.00_equilibration"
else
    printf -v j "%02d" $((${idx}-1))
    pstep="step6.${j}_equilibration"
    
    # create new mdin file
    perl -pe "s/${pstep}/${istep}/g; s/nstlim=[^,]*/nstlim=${nstlim}/" ${WINDOW}/${pstep}.mdin > ${WINDOW}/${istep}.mdin
fi

init="step3_pbcsetup"
SANDER="srun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"
echo "window: $WINDOW $istep $pstep"

###### JOB ######
cd $WINDOW

$SANDER -O -i ${istep}.mdin -o ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -r ${istep}.ncrst -x ${istep}.nc -ref ${init}.ncrst -inf ${istep}.mdinfo

date
