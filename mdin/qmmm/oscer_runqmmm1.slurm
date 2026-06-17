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

SANDER="mpiexec.hydra -bootstrap slurm -n ${SLURM_NTASKS} sander.MPI"

init="step3_pbcsetup"
istep="step5.00_equilibration"

if [ -f list ]; then
    total_windows=$(wc -l < list)
    if [ "${total_windows}" -eq 0 ]; then
        echo "missing n_windows" >&2
        exit 1
    fi
    n_windows=$((total_windows - 1))
else
    echo "missing n_windows" >&2
    exit 1
fi

for i in $(seq 0 "${n_windows}"); do
    printf -v window "%02d" $i
    
    cd $window
    $SANDER -O -i ${istep}.mdin -o ${istep}.mdout -p ${init}.parm7 -c ${istep}_inp.ncrst -r ${istep}.ncrst -x ${istep}.nc -inf ${istep}.mdinfo

    printf -v j "%02d" $(($i + 1))
    cp sinrvels.rst ../${j}/

    cd ..
done


date
