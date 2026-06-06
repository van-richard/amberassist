# Notes

## Requirements

### macOS (Apple silicon)

Install Lima with Homebrew:

```bash
brew install lima
```

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/).

### Linux (x86_64)

Install Docker Engine using the
[official instructions](https://docs.docker.com/engine/install/). For Ubuntu:

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation:
sudo docker run hello-world
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

