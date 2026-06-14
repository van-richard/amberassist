#!/bin/bash
#SBATCH -p batch
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -J reprocess
#SBATCH --array=1

set -euo pipefail
shopt -s nullglob

date

module load amber/23-panxl
module load qmhub

export MKL_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"
export OMP_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"

sander_cmd=(srun -n "${SLURM_NTASKS_PER_NODE}" sander.MPI)

init="step3_pbcsetup"
pstep="step5.00_equilibration"
istep="step7_reprocess"
trajin="step6"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "${script_dir}")"
window_list="${project_dir}/list"

[[ -f "${window_list}" ]] || { echo "ERROR: missing window list: ${window_list}"; exit 1; }

window="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${window_list}")"
[[ -n "${window}" ]] || { echo "ERROR: no window for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${window_list}"; exit 1; }
[[ "${window}" =~ ^[0-9][0-9]$ ]] || { echo "ERROR: unexpected window name: ${window}"; exit 1; }
[[ -d "${project_dir}/${window}" ]] || { echo "ERROR: missing window directory: ${project_dir}/${window}"; exit 1; }

echo "WINDOW ${window}"

cd "${project_dir}/${window}"

[[ -f "${init}.parm7" ]] || { echo "ERROR: missing topology: ${init}.parm7"; exit 1; }
[[ -f "${pstep}.ncrst" ]] || { echo "ERROR: missing restart: ${pstep}.ncrst"; exit 1; }

traj_files=( "${trajin}"*.nc )
(( ${#traj_files[@]} > 0 )) || { echo "ERROR: no trajectory files matching ${trajin}*.nc"; exit 1; }

cleanup() {
    [[ -L qmhub2.ini ]] && rm -f -- qmhub2.ini
}
trap cleanup EXIT

ln -sf "${script_dir}/qmhub2.ini" .
bash "${script_dir}/write_mdin.sh" "${script_dir}" "step7_reprocess.mdin" "${window}"

"${sander_cmd[@]}" -O -i "${istep}.mdin" -o "${istep}.mdout" -p "${init}.parm7" -c "${pstep}.ncrst" -y "${traj_files[@]}" -x "${istep}.nc"

# If reprocessing succeeds, run ./dedup.sh manually to compact duplicate qmmm.inp_* files.
ln -sf "${script_dir}/dedup.sh" .
# bash dedup.sh

# Do not create or replace qmhub.squashfs here without explicit user approval.
# module load base
# mksquashfs qmhub qmhub.squashfs

date
