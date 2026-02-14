# QM/MM Postprocessing with QMHub

This directory contains scripts and templates for **postprocessing QM/MM trajectories** from umbrella sampling simulations using the **QMHub–Amber interface**.

The purpose is to **re-evaluate saved trajectory frames** (e.g., from classical MD) with the QM/MM Hamiltonian to obtain consistent potential energy and force data for later analysis — such as **WHAM**, **machine learning potential training**, or **free-energy reconstruction**.

---

## Folder Contents

| File                          | Purpose                                                                |
| ----------------------------- | ---------------------------------------------------------------------- |
| `step7_reprocessing.mdin`     | Amber control file for single-point QM/MM energy reprocessing.         |
| `qmhub2.ini`                  | QMHub configuration file linking Amber MM region to the QM engine.     |
| `reprocessing.slurm`          | SLURM array script that submits all umbrella windows for reprocessing. |
| `dedup.sh`                    | Cleans up duplicate `qmmm.inp_*` files generated during QMHub runs.    |

---

## Workflow Overview

Each umbrella window is reprocessed independently through a SLURM job array.

1. **SLURM array submission**
   Each array task corresponds to a single umbrella window (`window_00`, `window_01`, …).

2. **Prepare the QM region**
   The QM region is defined using an Amber mask read from `qm.mask` or a per-window CSV (`masks.csv`).
   Atom indices are extracted automatically via `ambmask` or `cpptraj`:

   ```bash
   ambmask -p system.parm7 -c window_00.ncrst -mask "$(cat qm.mask)" -out atomnum > qm_mask.txt
   ```

3. **Generate input files**
   The `step7_reprocessing.mdin` template is expanded using `envsubst` to replace `${WIN_DIR}` with the current working directory.

4. **Run QM/MM reprocessing**
   Example execution command inside each job:

   ```bash
   srun -n $SLURM_NTASKS sander.MPI -O \
     -i step7_reprocessing.mdin \
     -p system.parm7 \
     -c window_${WIN}.ncrst \
     -y window_${WIN}.nc
   ```

5. **Cleanup**
   Duplicate QMHub input files (`qmmm.inp_####`) are compacted by running:

   ```bash
   bash reprocess/dedup.sh
   ```

   Outputs (`.mdout`, `qmhub/`) are then copied back to the submit directory.

---

## 🧠 Computational Notes

* Each reprocessing job performs **single-point QM/MM energy evaluations** (`imin=5`, `maxcyc=1`).

* **OpenMP and BLAS threading are disabled** for best scaling:

  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  ```

* **Parallelization strategy:**
  Jobs are distributed **across frames** rather than within a single sander run.
  Each SLURM task handles different frames or windows independently.

* **Performance tips:**

  * Use `--cpu-bind=cores --hint=nomultithread` in SLURM.
  * Stage input files to `$SLURM_TMPDIR` for faster I/O.
  * Disable periodic boundary conditions (`ntb=0`) for non-PBC reprocessing unless PME is required.
  * Reduce unnecessary output: `ntpr=0`, `ntwx=0`, `ntwr=0`.

---

## 🧩 QMHub Integration

`qmhub2.ini` defines how QMHub communicates between Amber and the QM engine.
Typical parameters include:

```ini
[simulation]
save_input = True

[model]
switching_function = lrec
cutoff = 10.0
swdist = 10.0
pbc = False

[engine]
dummy
```

During each frame reprocessing step, QMHub:

* Reads the QM atom list from `qm_mask.txt`.
* Extracts coordinates and MM point charges from Amber.
* Generates per-frame QM input files (`qmmm.inp_####`) used for external QM calculations or Δ-learning datasets.

---

## Quick Start

```bash
# Submit all 42 umbrella windows
sbatch reprocess/reprocessing.slurm

# Monitor progress
squeue -u $USER -n reproc

# Inspect output
less windows/00/step7_win00.mdout
```

---

## Maintenance Tips

* Clean up intermediate QMHub inputs:

  ```bash
  find . -name "qmmm.inp_*" -delete
  ```
* Verify QM region size:

  ```bash
  wc -l qmhub/qm_mask.txt
  ```
* Update `masks.csv` if the QM region changes across windows.

---

## Summary

This setup performs **efficient, automated postprocessing** of QM/MM trajectories with Amber and QMHub using SLURM arrays.
It is optimized for:

* Parallel frame re-evaluation
* Minimal I/O
* Consistent QM region definition

and provides reproducible QM/MM datasets for further mechanistic or machine-learning analysis.

