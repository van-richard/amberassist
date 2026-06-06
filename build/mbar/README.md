# Build

- Run this locally on MacOS. This creates `mbar_amd64.sif`, copy to HPC with `rsync`.

```bash
bash build_[machine]_amd64.sh # [machine] is your computer (mac or linux)
```

# Usage

```bash
apptainer exec -B /scratch:/scratch -W "$PWD" mbar_amd64.sif python mbar.py --step step6 
```

- this example uses [mbar.py](../../mbar/mbar.py)

```bash "use SIF to start SHELL"
apptainer shell -W "$PWD" mbar_amd64.sif 
```
