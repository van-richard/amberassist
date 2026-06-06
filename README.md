# amberassist

`amberassist` is a collection of workflow templates and helper utilities for
AMBER and AmberTools setup, classical molecular dynamics (MD), QM/MM
simulations, HPC job submission, free-energy analysis, QMHub reprocessing,
machine-learning potential experiments, and container builds. It is intended as
a practical research toolbox from which users can adapt selected files to their
own systems.

## Repository Status

This is a research workflow repository, not a polished or installable software
package. Some scripts and templates are broadly reusable, while others preserve
project-specific assumptions, exploratory analyses, or legacy material.

Treat the repository as a set of starting points. Review each file before use
and validate the resulting workflow for the target molecular system, software
versions, and computing environment.

## Repository Organization

See [`docs/repository-map.md`](docs/repository-map.md) for a status-oriented
guide to active utilities, reusable templates, project-specific examples,
legacy material, and files deferred for further review.

## Who This Is For

This repository may be useful to:

- computational chemistry and computational biochemistry researchers;
- AMBER and AmberTools users preparing or analyzing simulations;
- users running classical MD or QM/MM calculations;
- researchers working with MBAR, weighted thermodynamic perturbation (WTP), or
  related free-energy analyses; and
- users adapting simulation and analysis workflows to SLURM-based HPC systems.

## Directory Guide

| Path | Purpose | Status / notes |
| --- | --- | --- |
| `ambertools/` | Shell scripts, Python scripts, and notebooks for tools such as `tleap`, `cpptraj`, `pytraj`, `antechamber`, `parmchk2`, and ParmEd. | Mix of reusable helpers and system-specific analysis examples. |
| `build/` | Docker-based environments and scripts for producing container images, including Apptainer/SIF images for HPC use. | Host- and platform-dependent; read the local build documentation before running. |
| `fmatch/` | Force-matching and semi-empirical reparameterization experiments, including data conversion and model-evaluation helpers. | Research code with dataset- and model-specific assumptions. |
| `io/` | Input-generation, training-set, trajectory-reorganization, and SLURM helper scripts. | Many files assume particular directory layouts, cluster environments, or molecular systems. |
| `mbar/` | MBAR/PMF analysis code, notebooks, plotting helpers, and historical implementations. | Contains exploratory notebooks and a `legacy/` area; verify compatibility with the installed PyMBAR version. |
| `mdin/` | AMBER input templates for classical MD and QM/MM workflows, including umbrella-sampling-related files. | Templates contain scientific defaults that must be reviewed for each system. |
| `mlp/` | Machine-learning potential and delta-learning experiments, including TorchMD-Net-related files. | Experimental research workflows rather than a general training package. |
| `notebooks/` | Standalone exploratory and analysis notebooks. | Often project-specific and may require data not included in the repository. |
| `reprocess/` | Templates and scripts for QM/MM trajectory reprocessing through QMHub. | Designed around specific QMHub, AMBER, and HPC workflows; inspect local paths and configuration. |
| `util/` | Smaller utilities for structure and residue-related data preparation. | Review expected input formats and working-directory assumptions. |
| `wtp/` | Weighted thermodynamic perturbation analysis using QMHub/Q-Chem outputs, plus combination and PMF helpers. | Includes current examples, notebooks, and legacy scripts; consult `wtp/README.md` and verify filenames before use. |

## Common Workflows

### AMBER and AmberTools Templates

The `ambertools/` directory contains examples for topology and ligand setup,
trajectory processing, PCA, RMSD/RMSF analysis, and related tasks. Select only
the relevant helper, copy it into the working project if appropriate, and
review all filenames and selections before execution.

### MD and QM/MM Input Templates

Classical MD templates are under `mdin/md/`, while QM/MM and umbrella-sampling
templates are under `mdin/qmmm/`. These files are examples, not validated
defaults for arbitrary systems. Check simulation controls, atom masks,
restraints, topology names, coordinate names, and engine configuration.

### SLURM Job Templates

SLURM examples appear in `io/slurm/`, `mdin/`, `reprocess/`, and `wtp/`.
Before submission, update account and partition settings, resource requests,
module or environment setup, scratch paths, executable locations, and input
file paths for the target cluster.

### MBAR and Free-Energy Analysis

The `mbar/` and `mbar-sliding/` directories contain scripts and notebooks for
PMF estimation, plotting, and sliding-window analyses. Confirm the expected
collective-variable files, window definitions, force constants, sampling
intervals, and PyMBAR API version before using an analysis.

### WTP Analysis

The `wtp/` workflow consumes per-frame QM/MM inputs and combines results for
weighted thermodynamic perturbation and later PMF analysis. See
[`wtp/README.md`](wtp/README.md) for the intended data layout and theory
configuration. Verify that referenced scripts and filenames match the current
checkout before launching jobs.

### ML Potential and Delta-Learning Experiments

The `mlp/` and `fmatch/` directories contain experimental model-training,
conversion, force-matching, and evaluation code. These workflows assume
specific array shapes, atom counts, model formats, and datasets. They should be
treated as research examples rather than reusable model APIs.

### Container Builds for HPC Use

The `build/` directory contains Dockerfiles and host-side scripts for building
Linux images and converting or transferring them for Apptainer-based HPC use.
Follow [`build/README.md`](build/README.md) and the README in the relevant
subdirectory. Container builds may require Docker, Docker Buildx, Lima on
Apple-silicon macOS, and Apptainer.

## Installation and Dependencies

There is currently no package metadata or supported `pip install` workflow for
the repository as a whole. Clone the repository and use or adapt the files
needed for a specific workflow.

Dependencies vary by workflow. Depending on the selected scripts, they may
include:

- AMBER or AmberTools;
- Python 3 with NumPy, SciPy, and Pandas;
- PyMBAR and scikit-learn;
- `pytraj`, ParmEd, or other AmberTools Python interfaces;
- QMHub and a supported QM engine such as Q-Chem;
- PyTorch, PyTorch Lightning, or TorchMD-Net for ML experiments;
- a SLURM environment for HPC job templates; and
- Docker, Docker Buildx, Lima, or Apptainer for container workflows.

Check imports, executable calls, and local documentation for the selected
workflow rather than installing every dependency listed above.

## Usage Examples

Most workflows are intended to be adapted rather than run directly from the
repository root.

```bash
# Start from the relevant workflow directory.
cd mdin/qmmm

# Copy the required templates into a project working directory.
cp -R input /path/to/project/
```

Then:

1. inspect the copied templates and helper scripts;
2. edit system-specific paths, topology and coordinate filenames, atom masks,
   restraint definitions, and cluster settings;
3. run the files with the appropriate AmberTools, AMBER, QMHub, analysis, or
   scheduler commands on the target system; and
4. validate outputs on a small test case before launching production work.

For a documented container example:

```bash
cd build/ambertools23
bash build_mac_amd64.sh
```

Use the Linux build script instead when building on a compatible Linux host.

## Notes on Project-Specific Files

Some files contain historical or project-specific assumptions, including local
filesystem paths, cluster environment setup, directory naming conventions,
fixed atom counts, atom masks, restraint definitions, and SLURM resource
settings. Legacy directories preserve earlier workflow versions and should not
be assumed to be canonical.

Before using any template or script, inspect at least:

- input and output paths;
- topology, coordinate, trajectory, and model filenames;
- atom and residue masks;
- collective variables and restraint definitions;
- force constants and other simulation controls;
- QM method, basis, charge, and multiplicity settings; and
- SLURM account, partition, memory, task, and scratch configuration.

Do not assume that defaults from one research system are appropriate for
another.

## Development Notes

Minimal Ruff configuration is provided in `pyproject.toml` for Python static
checks. A repository-level pre-commit configuration is not currently provided.
See [`AGENTS.md`](AGENTS.md) for contribution and validation guidance,
especially the requirement to preserve scientific behavior unless a change is
explicitly requested.

Future cleanup should be performed in small, reviewable pull requests so that
repository hygiene, portability changes, and scientific behavior changes remain
separate.

## License

This repository is licensed under the GNU General Public License, version 3
(GPL-3.0). See [`LICENSE`](LICENSE) for the full license text.
