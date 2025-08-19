#!/bin/bash
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=05:00:00
#SBATCH --job-name=fmatch
#SBATCH --array=1

date

module load intel/2020a
source /home/panxl/qchem/trunk2/setqc.sh
source /home/panxl/amber20/amber.sh

# For QMHub only
eval "$(/home/panxl/.local/opt/miniforge3/bin/conda shell.bash hook)"
conda activate

export UCX_TLS=rc,sm,self
export I_MPI_PIN=0
export I_MPI_THREAD_YIELD=3
export I_MPI_THREAD_SLEEP=200
export I_MPI_OFI_PROVIDER=verbs

# For QMHub only
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

SANDER="mpiexec.hydra -bootstrap slurm -np 16 sander.MPI"

init="step3_pbcsetup"
pstep="step5.00_equilibration"
traj="step6_all"
istep="step7_reprocessing"

w=$(sed -n "$SLURM_ARRAY_TASK_ID"p list)
echo "WINDOW ${w}"


tmp="/scratch/van/b3lyp-pm3-sinr-newrxncoord"

cd ../${w}
ln -sf ../ml/traj.in .
cpptraj traj.in
ln -sf ../ml/dedup.sh .
ln -sf ../ml/qmhub2.ini .
sed "s/__WIN__/${w}/" ../ml/step7_reprocessing.mdin > step7_reprocessing.mdin

mkdir -p ${tmp}/${w}

$SANDER -O -i ${istep}.mdin -o  ${tmp}/${w}/${istep}.mdout -p ${init}.parm7 -c ${pstep}.ncrst -y ${traj}.nc -x ${tmp}/${w}/${istep}.nc

date
    	
mv ${tmp}/${w}/* .

mv /dev/shm/${w}qmhub qmhub

cd qmhub
bash dedup.sh

date

rm -r ${tmp}/${w}

date

