#!/usr/bin/bash
#SBATCH -J wtp
#SBATCH -p batch
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32         # one rank per core (adjust to node)
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --cpu-bind=cores
#SBATCH -o %J-%A_%a.out
#SBATCH -e %J-%A_%a.err
#SBATCH --array=0-41

date

set -euo pipefail

# --- Env (adjust to your site) ---
module load qmhub
# source ~/venvs/wtp/bin/activate

# Kill all threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WIN=${SLURM_ARRAY_TASK_ID}
ROOT="${SLURM_SUBMIT_DIR}"
WIN_DIR="${ROOT}/windows/${WIN}"
IN_QMH="${WIN_DIR}/qmhub"

# Stage lightweight code + inputs to local scratch
SCR="${SLURM_TMPDIR:-/tmp}/${USER}_wtp_${SLURM_JOB_ID}_${WIN}"
mkdir -p "${SCR}"/{qmh,code}
rsync -a "${IN_QMH}/" "${SCR}/qmh/"
rsync -a "${ROOT}/wtp/" "${SCR}/code/"   # brings wtp_energy.py, wtp_combine.py, theories.csv, mbar_pmf2.py
cd "${SCR}"

# Discover frames (qmmm.inp_????)
mapfile -t FRAMES < <(find qmh -maxdepth 1 -type f -name 'qmmm.inp_[0-9][0-9][0-9][0-9]' | sort)
N=${#FRAMES[@]}
(( N > 0 )) || { echo "No frames in ${IN_QMH}"; exit 1; }

# Read theories from CSV or env
declare -a TAGS METHODS BASES
if [[ -f code/theories.csv ]]; then
    while IFS=, read -r tag method basis; do
        [[ "$tag" == "tag" ]] && continue
        [[ -z "$tag" ]] && continue
        TAGS+=("$tag"); METHODS+=("$method"); BASES+=("$basis")
    done < code/theories.csv
elif [[ -n "${WTP_THEORIES:-}" ]]; then
    # Format: tag:method/basis,tag2:method/basis2
    IFS=, read -ra items <<< "$WTP_THEORIES"
    for it in "${items[@]}"; do
        tag="${it%%:*}"; mb="${it#*:}"
        method="${mb%%/*}"; basis="${mb#*/}"
        TAGS+=("$tag"); METHODS+=("$method"); BASES+=("$basis")
    done
else
    echo "No theories specified (theories.csv or WTP_THEORIES)."; exit 1
fi

T=${#TAGS[@]}
(( T > 0 )) || { echo "No theory rows parsed."; exit 1; }
echo "Window ${WIN}: ${N} frames; ${T} theories."

P=${SLURM_NTASKS}

# OPTIONAL: run multiple theories concurrently by splitting ranks
# If you prefer sequential per theory (simpler), set CONCURRENT=0
CONCURRENT=${CONCURRENT:-1}
if (( CONCURRENT == 1 )); then
    # Split ranks across theories
    RPTh=$(( P / T ))
    (( RPTh >= 1 )) || { echo "Too few ranks (${P}) for ${T} theories; set CONCURRENT=0."; RPTh=1; }
else
    RPTh=$P
fi

# Loop theories
for i in $(seq 0 $((T-1))); do
    tag="${TAGS[$i]}"; method="${METHODS[$i]}"; basis="${BASES[$i]}"
    OUT_DIR="${WIN_DIR}/wtp/${tag}"
    mkdir -p "${OUT_DIR}"

    echo ">>> WIN ${WIN} :: ${tag} :: ${method}/${basis}"

  # Fan-out across frames (per theory). Each rank handles every stride-th frame.
  if (( CONCURRENT == 1 )); then
      # Carve exclusive subset of ranks per theory
      # Start an srun job step with --ntasks=${RPTh} and a unique label
      srun --exclusive -N1 -n ${RPTh} bash -lc '
      set -euo pipefail
      r=$SLURM_PROCID
      RPTh='"$RPTh"'
      files=('"${FRAMES[@]}"')
      mkdir -p '"${tag}"'/frames
      for ((idx=r; idx<'"$N"'; idx+=RPTh)); do
          fin="${files[$idx]}"
          python3 code/wtp_energy.py \
              --inp "$fin" \
              --outdir '"${tag}"'/frames \
              --method "'"$method"'" \
              --basis "'"$basis"'"
          done
          ' &
      else
          # Sequential theory: use all ranks for this one theory
          srun --exclusive -N1 -n ${RPTh} bash -lc '
          set -euo pipefail
          r=$SLURM_PROCID
          RPTh='"$RPTh"'
          files=('"${FRAMES[@]}"')
          mkdir -p '"${tag}"'/frames
          for ((idx=r; idx<'"$N"'; idx+=RPTh)); do
              fin="${files[$idx]}"
              python3 code/wtp_energy.py \
                  --inp "$fin" \
                  --outdir '"${tag}"'/frames \
                  --method "'"$method"'" \
                  --basis "'"$basis"'"
              done
              '
  fi
done

# Wait for any concurrent theory steps
wait

# Combine per theory (per-window)
for i in $(seq 0 $((T-1))); do
    tag="${TAGS[$i]}"
    python3 code/wtp_combine.py --frames-dir "${tag}/frames" --outdir "${tag}"
done

# Sync all theory outputs home
for i in $(seq 0 $((T-1))); do
    tag="${TAGS[$i]}"
    rsync -a "${tag}/" "${WIN_DIR}/wtp/${tag}/"
done

echo "WTP window ${WIN} done."

