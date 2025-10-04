# Build

- Run this locally on MacOS. This creates `miniforge3_amd64.sif`, copy to HPC with `rsync`.

```bash
bash build_amd64.sh
```

# Usage

```bash
apptainer exec -B /scratch:/scratch -W "$PWD" miniforge3_amd64.sif 
```
