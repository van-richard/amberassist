# Repository Guidelines

## Scope

This repository contains reusable AMBER, QM/MM, MBAR, machine-learning, and
HPC workflow templates. Many scripts are project-specific examples rather than
an installed Python package.

## Change Guidelines

- Preserve scientific parameters, atom selections, force constants, units, and
  numerical behavior unless a task explicitly requires changing them.
- Treat files under `legacy/` as historical references. Do not modernize or
  remove them without confirming the canonical replacement.
- Do not delete tracked topology, restart, trajectory, NumPy, or notebook data
  solely because it resembles generated output. Confirm whether it is a fixture
  or required example first.
- Prefer command-line arguments or environment variables over personal
  `/home`, `/Users`, and `/scratch` paths in new scripts.
- Keep cluster-specific setup in environment files or documented variables.
- Avoid running production SLURM, AMBER, QMHub, Q-Chem, or model-training jobs
  as part of routine validation.
  - Prefer small, reviewable diffs.
- Do not change scientific equations, units, force constants, atom masks, or simulation defaults unless explicitly asked.
- Treat files under `legacy/` as archival unless the task explicitly targets them.
- Preserve current command-line behavior when refactoring scripts.
- Replace hard-coded local paths with CLI arguments or environment variables.
- Add helpful error messages for missing files, missing executables, and invalid paths.
- Do not commit generated data, trajectories, restart files, container images, or notebook checkpoints.

## Validation

For repository hygiene changes, run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' .
ruff check .
bash -n path/to/changed-script.sh
```

Use targeted checks for changed workflows. Large scientific calculations
require explicit user direction and the appropriate external software.

For Python changes:
- Run `ruff check .` when available.
- Run targeted tests if present.
- At minimum, run changed scripts with `--help` after adding argparse.

For Docker/container changes:
- Do not bump scientific package versions casually.
- Keep AmberTools, PyMBAR, NumPy, SciPy, and Python versions explicit.
- Document whether the image targets macOS build hosts, Linux hosts, or HPC Apptainer use.

## Style

- Use `pathlib.Path` for filesystem paths.
- Prefer `argparse` for standalone scripts.
- Keep reusable logic inside functions.
- Put execution behind `if __name__ == "__main__":`.

