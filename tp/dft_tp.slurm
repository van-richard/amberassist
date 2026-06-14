#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j-TP.out
#SBATCH --error=%j-TP.err
#SBATCH --job-name=dft-tp
#SBATCH --array=10-42

set -euo pipefail

date

module load amber/23-panxl
module load qchem/5.2
module load qmhub

export MKL_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"
export OMP_NUM_THREADS="${SLURM_NTASKS_PER_NODE}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

method="wb97xd"
basis="6-31+gd"
case "${basis}" in
    "6-31+g*")
        basis="6-31+gd"
        ;;
    "6-31g*")
        basis="6-31gd"
        ;;
esac
name="${method}_${basis}"
py="dft_tp"
strt="0"
end="498"
iter="2"
overwrite="0"

window="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${root_dir}/list")"
if [[ -z "${window}" ]]; then
    echo "No window found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "window ${window}"
echo "method ${method}"
echo "basis ${basis}"
echo "overwrite ${overwrite}"

for ((i = strt; i <= end; i += iter)); do
    printf -v j "%04d" "${i}"

    input_file="${root_dir}/${window}/qmhub/qmmm.inp_${j}"
    output_dir="${script_dir}/qmmm_energies/${name}/${window}/${j}"
    energy_file="${output_dir}/qmmm_${name}_energy.npy"
    forces_file="${output_dir}/qmmm_${name}_forces.npy"

    if [[ ! -f "${input_file}" ]]; then
        echo "Missing unpacked QMHub input: ${input_file}" >&2
        echo "Expected an existing qmhub/ directory; qmhub.squashfs is not unpacked by this job." >&2
        exit 1
    fi

    if [[ "${overwrite}" != "1" ]]; then
        if [[ -f "${energy_file}" && -f "${forces_file}" ]]; then
            echo "Skipping complete frame ${window}/${j}"
            continue
        fi
        if [[ -f "${energy_file}" || -f "${forces_file}" ]]; then
            echo "Partial outputs exist for ${window}/${j}; set overwrite=1 after checking the files." >&2
            exit 1
        fi
    fi

    mkdir -p "${output_dir}"

    args=(
        python -u "${script_dir}/${py}.py"
        "${input_file}"
        "${method}"
        "${basis}"
        "${output_dir}"
        "qmmm_${name}"
    )
    if [[ "${overwrite}" == "1" ]]; then
        args+=(--force)
    fi

    "${args[@]}"
    echo "Finished ${window}/${j}"
done

date
