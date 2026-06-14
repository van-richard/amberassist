# Repository Map

This map describes the intended status of major repository areas. Categories
are navigational labels, not guarantees that a workflow is portable or
scientifically appropriate for a new system.

| Path | Category | Purpose | Notes / cautions |
| --- | --- | --- | --- |
| `ambertools/` | template | AmberTools setup and analysis helpers for `tleap`, `cpptraj`, `pytraj`, ligand preparation, and related tasks. | Contains both reusable shell helpers and project-shaped notebooks/scripts. Inspect filenames, masks, and trajectory assumptions. |
| `build/` | template | Docker and Apptainer/SIF build workflows for AmberTools, MBAR, Miniforge, QMHub, and related environments. | Build-host and architecture assumptions vary by subdirectory. Generated images are not source material. |
| `eda/` | template | QM/MM energy decomposition analysis for QM/MM, gas-phase, PCM, and Lennard-Jones force terms, mean-force PMFs, and residue-level decomposition. | Active workflow with strong assumptions about `../00` through `../41`, visible QMHub inputs, per-window `eda/` outputs, and cache regeneration. |
| `examples/` | example | Explicitly project-specific workflows separated from general helpers. | Examples preserve their original scientific assumptions and are not generic defaults. |
| `fmatch/` | example | Force-matching, parameter perturbation, data conversion, and model-evaluation experiments. | Assumes particular datasets, atom counts, model formats, and external modules. |
| `io/` | unclear | Input generation, training-set preparation, trajectory reorganization, and SLURM helpers. | Mixed-purpose area with reusable utilities and cluster/project-specific scripts. Review individual files before reuse. |
| `mbar/` | template | MBAR/PMF analysis utilities, notebooks, plotting, and historical implementations. | Current and exploratory material coexist. `mbar/legacy/` is archival; notebooks may embed local paths or saved outputs. |
| `mbar-sliding/` | example | Sliding-window MBAR analysis and plotting workflow. | Appears reusable in structure but retains workflow-specific file and sampling assumptions. |
| `mdin/` | template | Classical MD and QM/MM AMBER input templates, restraints, and related run files. | Scientific defaults, masks, restraint values, and paths are system-specific and must be reviewed without assuming portability. |
| `mlp/` | example | Machine-learning potential and delta-learning experiments. | Research code tied to specific datasets, tensor shapes, dependencies, and training procedures. |
| `notebooks/` | example | Exploratory analysis notebooks. | May require data not present in the repository and may contain project history or stored outputs. |
| `reprocess/` | template | AMBER/QMHub trajectory reprocessing templates, including SLURM array submission, mdin generation, QMHub configuration, and duplicate-input cleanup. | Active templates coexist with archived versions. Review window lists, `qm_info.txt` or mdin metadata, `qmhub/` outputs, and manual deduplication before running. |
| `tp/` | template | DFT thermodynamic perturbation workflow for QMHub/Q-Chem single-point calculations, frame energy/force outputs, combined TP arrays, and plotting. | Requires existing unpacked `../WINDOW/qmhub/qmmm.inp_????` inputs and external QM software; writes TP outputs under `tp/qmmm_energies/` and `mbar/tp_energy/`. |
| `util/` | active | Small structure and residue-data utilities. | More general-purpose than most areas, but expected inputs and working-directory behavior still need checking. |
| `wtp/` | template | Weighted thermodynamic perturbation, result combination, and PMF-related helpers. | Active examples coexist with notebooks and `wtp/legacy/`; verify current filenames and external QM dependencies. |

## Category Meanings

- **active**: relatively general-purpose utility intended for current use.
- **template**: current workflow material intended to be copied and adapted.
- **example**: research or project-specific material retained as a worked
  example.
- **legacy**: historical material retained for reference, not the preferred
  starting point.
- **unclear**: mixed or uncertain ownership that needs domain review before
  relocation.

## Explicit Legacy Areas

The following directories are preserved as archives:

- `mbar/legacy/`
- `mdin/qmmm/legacy/`
- `eda/legacy/`
- `reprocess/legacy/`
- `wtp/legacy/`

Prefer material outside these directories unless reproducing or comparing an
older workflow.

## Deferred Review

The following files or groups were intentionally left in place because moving
them safely requires scientific or workflow-owner review:

| Path | Reason deferred |
| --- | --- |
| `ambertools/pytraj/pca-Copy2.ipynb` | Copy-like name and stored project output suggest archival status, but the canonical notebook is uncertain. |
| `ambertools/pytraj/*.py` and `ambertools/pytraj/*.ipynb` | Several are notebook exports or project-shaped analyses; relationships between script and notebook versions need review. |
| `fmatch/` | Hard-coded paths and dataset assumptions indicate example status, but individual files may form one coupled workflow. |
| `io/` except the moved aldol example | Contains hard-coded cluster paths, versioned filenames, and potentially coupled SLURM/script workflows. |
| `mbar/notebooks/` and `mbar/mbar.ipynb` | Notebooks contain local paths and historical outputs, but may still document active analyses. |
| `mdin/qmmm/asm/STOP_STRING.BAK` and `mdin/qmmm/asm/STRING.BAK` | Backup-like names, but their role in the ASM workflow is not established. |
| `mdin/qmmm/asm/` | Contains system-specific paths and generated-looking inputs that may be required reference data. |
| `mlp/` | Clearly experimental, but file relationships and expected working directories should be reviewed before finer reorganization. |
| `notebooks/` | Project-specific by nature, but insufficient evidence exists to assign notebooks to individual workflow owners. |

No files in this deferred list should be moved or removed solely based on their
names or hard-coded paths.
