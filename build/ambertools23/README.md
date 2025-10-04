# Build

<<<<<<< HEAD:build/ambertools23/README.md
Run this locally on MacOS. This creates `mbar_amd64.sif`, copy to HPC with `rsync`.

```bash
bash build_[machine]_amd64.sh # [machine] is your computer (mac or linux)
=======
- Run this locally on MacOS. This creates `mbar_amd64.sif`, copy to HPC with `rsync`.

```bash
bash build_amd64.sh
>>>>>>> test:build/ambertools/README.md
```

# Usage

<<<<<<< HEAD:build/ambertools23/README.md
```bash "use SIF to run python scripts"
apptainer exec -B /scratch:/scratch -W "$PWD" ambertools23_amd64.sif python mbar.py --step step6 
```

- this example uses [mbar.py](../../mbar/mbar.py)

```bash "use SIF to start SHELL"
apptainer shell -W "$PWD" ambertools23_amd64.sif 
```
=======
```bash
apptainer exec -B /scratch:/scratch -W "$PWD" ambertools24_amd64.sif [tool]
```

- where `[tool]` can be `tleap`, `cpptraj`, `parmchk2`, etc.
>>>>>>> test:build/ambertools/README.md
