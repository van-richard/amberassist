#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j
#SBATCH --error=%j
#SBATCH --job-name=wtp  
#SBATCH --array=1--42

date

# # For Pete  
# module load amber/23-panxl
# module load qmhub
# SANDER="mpirun -n ${SLURM_NTASKS_PER_NODE} sander.MPI"

# For Oscer
source /home/van/oscer.qmmm.env
SANDER="mpiexec.hydra -bootstrap slurm -n 1 -ppn 16 sander.MPI"

export MKL_NUM_THREADS=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p ../list)
echo "window $LINE"

window=$LINE
name="wtp"

# frame range
strt="0"
end="500"
iter="2"

cd ../${window}
mkdir -p $name # make new work directory
cd $name

seq $strt $iter $end | while read i; do 
	mkdir -p $i 
	cd $i
	printf -v j "%04d" $i
	python -u wtp_energy.py ../../qmhub/qmmm.inp_${j}
	echo "Finished $i"
	cd -
done

date

