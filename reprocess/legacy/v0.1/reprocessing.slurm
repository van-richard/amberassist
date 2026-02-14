#!/usr/bin/bash
#SBATCH -J step7
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32          # one rank per core (adjust to your node)
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --cpu-bind=cores
#SBATCH -o %J-%A_%a.out
#SBATCH -e %J-%A_%a.err
#SBATCH --array=0-41

set -euo pipefail
module load amber/23                   # adjust to your site

# 0) Turn off all threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WIN=${SLURM_ARRAY_TASK_ID}
PRMTOP="step3_pbcsetup.parm7"
COORD="${WIN}/step5.00_equilibration_inp.ncrst"
TRAJ="${WIN}/step6.??_equilibration.nc"

# 1) Stage to node-local scratch (faster FS)
WDIR="/tmp/${USER}/reproc_${SLURM_JOB_ID}_${WIN}"
mkdir -p "$WDIR" && rsync -a "$PRMTOP" "$COORD" "$TRAJ" reprocess/ "$WDIR"/
cd "$WDIR"

# 2) Build QM mask from Amber mask string (or use your CSV approach)
QMSTR=$(<../input/qm.mask)   # one-line Amber mask
ambmask -p "$PRMTOP" -c "$COORD" -mask "$QMSTR" -out atomnum > qm_mask_1based.txt
awk '{for(i=1;i<=NF;i++) printf("%d\n",$i-1)}' qm_mask_1based.txt > qmhub/qm_mask.txt

# 3) Create mdin from template (envsubst keeps it simple)
export WIN_DIR="$PWD"
envsubst < reprocess/step7_reprocessing.mdin > mdin.in

# 4) Discover frame count
#    (cpptraj prints # of frames without loading them all)
nframes=$(cpptraj -p "$PRMTOP" -y "$TRAJ" <<<'run' 2>/dev/null | awk '/Frames in trajectory:/ {print $4; exit}')
: "${nframes:?Failed to count frames}"

# 5) Parallelize across frames with srun ranks (offset/stride pattern)
P=${SLURM_NTASKS}      # total ranks on this node
srun bash -lc '
  set -euo pipefail
  r=$SLURM_PROCID
  P='"$P"'
  n='"$nframes"'
  for ((f=r; f<n; f+=P)); do
    # materialize qmmm.inp for frame f (on the fly, no big files)
    python3 reprocess/materialize.py \
       --container "" \            # if using your compact container; else remove
       --prmtop "'"$PRMTOP"'" \
       --traj "'"$TRAJ"'" \
       --frame "$f" \
       --out qmhub/qmmm.inp_$(printf "%04d" "$f") 2>/dev/null || true

    # run sander for this frame (single-rank)
    sander -O \
      -i mdin.in \
      -o step7_win'"$WIN"'_f$(printf "%06d" "$f").mdout \
      -p "'"$PRMTOP"'" \
      -c "'"$COORD"'" \
      -y "'"$TRAJ"'" \
      >/dev/null 2>&1
  done
'

# 6) Copy results back
rsync -a --ignore-missing-args *.mdout qmhub/ "$SLURM_SUBMIT_DIR"/windows/${WIN}/
