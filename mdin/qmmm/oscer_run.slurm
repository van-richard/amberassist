#!/bin/bash
#SBATCH -p normal
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %jrunqmmm1.out
#SBATCH -e %jrunqmmm1.err

date

module load intel/2020a 
source /home/panxl/qchem/trunk2/setqc.sh
source /home/panxl/amber20/amber.sh
export UCX_TLS=all

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate
export I_MPI_PIN=0
export I_MPI_THREAD_YIELD=3
export I_MPI_THREAD_SLEEP=200
export I_MPI_OFI_PROVIDER=verbs

# For QMHub only
export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

#unset I_MPI_PMI_LIBRARY
SANDER="mpiexec.hydra -bootstrap slurm -n ${SLURM_NTASKS} sander.MPI"

init="step3_pbcsetup"

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

echo "window: $WINDOW $istep $pstep"

###### JOB ######
cd $WINDOW

$SANDER -O -i ${istep}.mdin -o ${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -r ${istep}.ncrst -x ${istep}.nc -ref ${init}.ncrst -inf ${istep}.mdinfo

date
