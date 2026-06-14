#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=/home/van/system/eda/lj_eda_%A_%a.out
#SBATCH --error=/home/van/system/eda/lj_eda_%A_%a.err
#SBATCH --job-name=lj_eda
#SBATCH --array=1-42

set -euo pipefail

date

module load amber/23-panxl

export MKL_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"
export OMP_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hostname

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$script_dir/../list")
echo "window $LINE"

export EDA_OUTPUT_DIR="$script_dir/../$LINE/eda"
mkdir -p "$EDA_OUTPUT_DIR"

cd "$script_dir/../$LINE"

python -u "$script_dir/lj_eda.py"

date
