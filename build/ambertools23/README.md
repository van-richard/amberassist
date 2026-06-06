# Build

Run the script for the local platform. It creates `ambertools24_amd64.sif`,
which can be copied to the HPC system with `rsync`.

```bash
bash build_[machine]_amd64.sh # [machine] is your computer (mac or linux)
```

# Usage

```bash
apptainer exec -B /scratch:/scratch -W "$PWD" ambertools24_amd64.sif [tool]
```

`[tool]` can be `tleap`, `cpptraj`, `parmchk2`, or another AmberTools command.

To start an interactive shell:

```bash
apptainer shell -B /scratch:/scratch -W "$PWD" ambertools24_amd64.sif
```
