# Build

Run this locally on MacOS. This creates `mbar_amd64.sif`, copy to HPC with `rsync`.

```bash
bash build_amd64.sh
```

# Usage

```bash
apptainer exec -B /scratch:/scratch -W "$PWD" mbar_amd64.sif python mbar.py --step step6 
```
