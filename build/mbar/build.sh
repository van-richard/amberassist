#!/usr/bin/env bash
set -euo pipefail

# ---------- config (edit if you like) ----------
IMG_TAG=${IMG_TAG:-mbar:amd64}
TAR_NAME=${TAR_NAME:-mbar_amd64.tar}
SIF_NAME=${SIF_NAME:-mbar_amd64.sif}

# lima instance name (what `limactl ls` shows). default matches your earlier messages.
LIMA_INSTANCE=${LIMA_INSTANCE:-apptainer}

# where to stage files inside the VM
REMOTE_DIR=/tmp
REMOTE_TAR="${REMOTE_DIR}/${TAR_NAME}"
REMOTE_SIF="${REMOTE_DIR}/${SIF_NAME}"

# ---------- sanity checks ----------
command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }
command -v limactl >/dev/null 2>&1 || { echo "limactl not found"; exit 1; }

echo "==> Step 1: build linux/amd64 image and export docker-archive tar"
docker buildx ls >/dev/null 2>&1 || { echo "Docker Buildx not available"; exit 1; }

# Build x86_64 image on Apple Silicon via QEMU
docker buildx build --platform linux/amd64 -t "${IMG_TAG}" .

# Export to docker-archive (most compatible for Apptainer)
docker save "${IMG_TAG}" -o "${TAR_NAME}"
echo "   created ${TAR_NAME}"

echo "==> Step 2: copy tar into Lima VM (${LIMA_INSTANCE}) at ${REMOTE_TAR}"
limactl copy "${TAR_NAME}" "${LIMA_INSTANCE}:${REMOTE_TAR}"

echo "==> Step 3a: convert tar -> SIF inside Lima with --arch amd64"
# run non-interactively inside the VM
limactl shell "${LIMA_INSTANCE}" -- bash -lc "
set -euo pipefail
cd '${REMOTE_DIR}'
# force amd64 manifest selection (belt + suspenders)
export APPTAINER_DOCKER_ARCH=amd64
export SINGULARITY_DOCKER_ARCH=amd64
apptainer build --force --arch amd64 '${SIF_NAME}' docker-archive://'${TAR_NAME}'
echo '   built ${SIF_NAME} in ${REMOTE_DIR}'
"

echo "==> Step 4: copy SIF back to macOS"
limactl copy "${LIMA_INSTANCE}:${REMOTE_SIF}" "./${SIF_NAME}"

echo "==> Done."
echo "SIF: $(pwd)/${SIF_NAME}"
echo
echo "Quick test (on macOS, requires apptainer installed locally):"
echo "  apptainer exec ${SIF_NAME} python -c \"import platform; print(platform.machine())\""
echo "  apptainer exec ${SIF_NAME} python -c \"import pymbar, numpy; print('OK', pymbar.__version__, numpy.__version__)\""

