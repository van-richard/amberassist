# Notes

Requirements (MacOS):
    - install `lima`

```bash
brew install lima
```

- install `docker`

```bash
https://docs.docker.com/desktop/?_gl=1*2oezum*_gcl_au*MTc3MTUzMTQ5My4xNzU1NzE2NDg4*_ga*MTI2MDE2NjAxNC4xNzU1NzE2NDc1*_ga_XJWPQMJYHQ*czE3NTU3MjU1NDUkbzIkZzEkdDE3NTU3MjU1NTUkajUwJGwwJGgw
```

---

## Workflow

```bash
 ┌────────────────────────────┐
 │   macOS (M2 Pro, Docker)   │
 └────────────────────────────┘
             │
             │  (1) build amd64 Docker image
             │
             ▼
     docker buildx build --platform linux/amd64 -t mbar:amd64 .
             │
             │
             │  (2) export image as tarball
             ▼
     docker save mbar:amd64 -o mbar_amd64.tar
             │
             │
             │  (3) copy tar into Lima VM
             ▼
 ┌────────────────────────────┐
 │   Lima VM (Linux arm64)    │
 │   with Apptainer installed │
 └────────────────────────────┘
             │
             │  limactl copy mbar_amd64.tar apptainer:/tmp/
             │
             │
             │  (4) convert tar → SIF
             ▼
     apptainer build --arch amd64 mbar_amd64.sif \
        docker-archive:///tmp/mbar_amd64.tar
             │
             │
             │  (5) copy finished SIF back out
             ▼
 ┌────────────────────────────┐
 │   macOS host filesystem    │
 │   (mbar_amd64.sif)         │
 └────────────────────────────┘
             │
             │  (6) scp/rsync SIF to cluster
             ▼
 ┌────────────────────────────┐
 │   Linux HPC (x86_64)       │
 │   Run with Apptainer       │
 └────────────────────────────┘

```

---

## Using `lima`

```bash
limactl create --name=apptainer template://docker
```

```bash
limactl start apptainer
```

```bash
limactl stop apptainer
```

---

## Building

1. Build SIF locally on MacOS (silicon)

```bash
bash build_[OS].sh
```

