# QM/MM Energy Decomposition Analysis

This folder contains post-processing workflows for QM/MM umbrella-sampling
free-energy profile energy decomposition analysis. The active workflow builds
per-window QM/MM, gas-phase, PCM, and Lennard-Jones force components, then uses
those arrays to compute mean-force PMFs and residue-level decompositions.

Window data live outside this folder in `../00` through `../41`. The EDA
producer scripts are launched from each window by Slurm and write outputs into
that window's `eda/` directory. Reference window data, trajectories, and packed
`qmhub.squashfs` files should not be modified or unpacked by this workflow.

## Workflow Overview

1. Make sure per-window QMHub inputs are visible as
   `../WINDOW/qmhub/qmmm.inp_????`.
   The EDA scripts read those files in place and do not unpack `qmhub.squashfs`.

2. Generate per-window EDA force outputs:
   - `qmmm_eda.slurm` runs `qmmm_eda.py` for the QM/MM reference calculation.
   - `gas_eda.slurm` runs `gas_eda.py` with MM charges zeroed for the gas
     reference.
   - `pcm_eda.slurm` runs `pcm_eda.py` with Q-Chem PCM enabled.
   - `lj_eda.slurm` runs `lj_eda.py` for Lennard-Jones force decomposition.

3. Generate coordinate caches:
   - `pos_all.py` builds full `pos_all.npy` coordinates for the three QM/CV
     atoms from visible `qmmm.inp_????` files.
   - `qm_pos.py` truncates `pos_all.npy` into `qm_pos.npy` using
     `[:42, :500:2, :, :]`. `qm_pos.npy` is the coordinate cache used by the
     analysis workflows.

4. Generate residue-force caches when needed:
   - `write_res_forces.py` reads per-window EDA arrays and writes folder-level
     residue/QM force caches such as `qmmm_res_forces2.npy`,
     `gas_res_forces2.npy`, and `vdw_res_forces2.npy`.

5. Run or inspect analysis workflows:
   - `mf.py` and `mf.ipynb` compute mean-force PMFs and decomposition terms.
   - `res_forces_eda.ipynb` analyzes residue-level EDA results.
   - `res_forces_eda_pcm.ipynb` is the PCM-aware residue workflow.

## Active Scripts

`qmmm_eda.py`
: Reads `qmhub/qmmm.inp_????` from a window directory and writes
  `qmmm_energy.npy`, `qmmm_forces.npy`, `qmmm_prot_forces.npy`,
  `qmmm_near_forces.npy`, `qmmm_far_forces.npy`, and `qmmm_mm_esp.npy` into
  `EDA_OUTPUT_DIR`, normally `../WINDOW/eda`.

`gas_eda.py`
: Mirrors `qmmm_eda.py`, but zeroes MM atom charges before collecting gas-phase
  reference force terms. Outputs are named `gas_*.npy`.

`pcm_eda.py`
: Mirrors the gas/QM/MM EDA workflow while keeping Q-Chem
  `solvent_method = pcm`. Outputs are named `pcm_*.npy`.

`lj_eda.py`
: Reads `step7_reprocessing.nc` in each window and computes Lennard-Jones force
  components. It writes `lj_prot_forces.npy`, `lj_near_forces.npy`, and
  `lj_prot_forces_sum.npy` into the window `eda/` directory.

`lj_eda_sum.py`
: Utility that rebuilds `lj_prot_forces_sum.npy` from `lj_prot_forces.npy`
  inside `EDA_OUTPUT_DIR`.

`pos_all.py`
: Builds the full coordinate cache `pos_all.npy` from visible QMHub input
  coordinates. It refuses to overwrite existing output unless `--force` is
  used.

`qm_pos.py`
: Builds the truncated analysis coordinate cache `qm_pos.npy` from
  `pos_all.npy`. It refuses to overwrite existing output unless `--force` is
  used.

`write_res_forces.py`
: Converts per-window EDA arrays into folder-level residue/QM caches used by
  residue analysis workflows.

`mf.py`
: Script form of the main mean-force analysis. It reads per-window `../WINDOW/eda`
  outputs and `qm_pos.npy`, computes PMFs and decomposition terms, and writes
  downstream tables/atom-level products when executed.

## Slurm Entry Points

The Slurm scripts are the intended way to run per-window producers:

```bash
sbatch qmmm_eda.slurm
sbatch gas_eda.slurm
sbatch pcm_eda.slurm
sbatch lj_eda.slurm
```

Each Slurm script reads `../list`, changes into the selected window directory,
sets `EDA_OUTPUT_DIR="$script_dir/../$LINE/eda"`, and runs the corresponding
Python script from this folder. This keeps generated per-window `.npy` files out
of the main simulation directory and inside `../WINDOW/eda`.

## Analysis Notebooks

`mf.ipynb`
: Interactive version of the main PMF/decomposition workflow. It currently
  keeps the PCM path active and loads `../WINDOW/eda/pcm_forces.npy`.

`res_forces_eda.ipynb`
: Residue-level decomposition workflow using `qm_pos.npy` and the residue force
  caches produced by `write_res_forces.py`.

`res_forces_eda_pcm.ipynb`
: PCM-aware residue-level workflow.

`write_res_forces.ipynb`
: Notebook version of `write_res_forces.py`. Prefer the Python script for
  repeatable cache generation.

`check_index_mmbarrier.ipynb`
: Auxiliary index-check notebook for older atom-level MM barrier products. It
  treats `barrier_mm.npy` or `barrier_mm.dat` as input and writes mapped,
  charge-normalized `mm_barrier_pot2.npy`. It is not part of the primary EDA
  workflow.

## Key Intermediate Files

Per-window outputs under `../WINDOW/eda`:

- `qmmm_*.npy`: QM/MM force and energy decomposition terms.
- `gas_*.npy`: gas-reference force and energy decomposition terms.
- `pcm_*.npy`: PCM-reference force and energy decomposition terms.
- `lj_prot_forces.npy`: atom-level protein LJ forces.
- `lj_near_forces.npy`: near-field LJ forces.
- `lj_prot_forces_sum.npy`: LJ protein force summed over protein atoms.

Folder-level caches:

- `pos_all.npy`: full three-atom QM/CV coordinate cache.
- `qm_pos.npy`: truncated coordinate cache used by `mf.py`, `mf.ipynb`, and
  residue workflows.
- `qmmm_qm_forces.npy` and `gas_qm_forces.npy`: QM atom force caches.
- `qmmm_res_forces2.npy`, `gas_res_forces2.npy`, and `vdw_res_forces2.npy`:
  residue-level force caches.
- `res_names.npy`: residue labels matching the residue force cache dimension.
- `barrier_res_table.pkl`: residue-level decomposition table from prior
  analysis.

`lj_prot_forces.npy` and `lj_prot_forces_sum.npy` have different roles.
Residue-level decomposition needs `lj_prot_forces.npy` because it keeps the
atom dimension. Total protein vdW PMF terms use `lj_prot_forces_sum.npy`
because the protein atom dimension has already been summed.

## Auxiliary / Historical Artifacts

`lj_prot_energy.py`
: Computes scalar LJ energies and is not used by the current force-based MF
  workflow.

`barrier_mm.npy`
: A compact copy of older `barrier_mm.dat` atom-level MM barrier data. Treat it
  as an input to the index-check notebook, not as an active output of `mf.py`.

`mm_barrier_pot.npy`
: Charge-normalized atom-level product from `mf.py` when that atom-level block
  is executed.

`mm_barrier_pot2.npy`
: Mapped, charge-normalized atom-level product from `check_index_mmbarrier.ipynb`.

`legacy/`
: Historical copies and notes. Do not use this directory as part of the active
  workflow unless explicitly comparing old behavior.

## Safety Notes

- Do not unpack, overwrite, or regenerate `qmhub.squashfs` from this workflow.
- Do not modify `../00` through `../41`; they are reference window data.
- Do not overwrite `.npy` caches unless intentionally regenerating them.
- Use `--force` on `pos_all.py` or `qm_pos.py` only when overwriting coordinate
  caches is intended.
- Run only lightweight validation unless intentionally launching the full
  Slurm workflows.
