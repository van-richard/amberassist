#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
IMG_TAG=${IMG_TAG:-mbar:amd64}
TAR_NAME=${TAR_NAME:-mbar_amd64.tar}
SIF_NAME=${SIF_NAME:-mbar_amd64.sif}
LIMA_INSTANCE=${LIMA_INSTANCE:-apptainer}
REMOTE_DIR=/tmp
REMOTE_TAR="${REMOTE_DIR}/${TAR_NAME}"
REMOTE_SIF="${REMOTE_DIR}/${SIF_NAME}"

# ---------- checks ----------
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v limactl >/dev/null || { echo "limactl not found"; exit 1; }
docker buildx ls >/dev/null || { echo "Docker Buildx not available"; exit 1; }

echo "==> Step 1: Build linux/amd64 image (QEMU) and export docker-archive tar"
docker buildx build --platform linux/amd64 -t "${IMG_TAG}" .
docker save "${IMG_TAG}" -o "${TAR_NAME}"
echo "   created ${TAR_NAME}"

echo "==> Step 2: Copy tar into Lima VM (${LIMA_INSTANCE}) at ${REMOTE_TAR}"
limactl copy "${TAR_NAME}" "${LIMA_INSTANCE}:${REMOTE_TAR}"

echo "==> Step 3: Convert tar -> SIF inside Lima (force amd64), non-interactive"
limactl shell "${LIMA_INSTANCE}" -- bash -lc "
set -euo pipefail
cd '${REMOTE_DIR}'
export APPTAINER_DOCKER_ARCH=amd64
export SINGULARITY_DOCKER_ARCH=amd64
apptainer build --force --arch amd64 '${SIF_NAME}' docker-archive://'${TAR_NAME}'
echo '   built ${SIF_NAME} in ${REMOTE_DIR}'
"

echo "==> Step 4: Copy SIF back to macOS"
limactl copy "${LIMA_INSTANCE}:${REMOTE_SIF}" "./${SIF_NAME}"

echo "==> Done."
echo "SIF: $(pwd)/${SIF_NAME}"
echo
echo "Quick test (needs apptainer locally):"
echo "  apptainer exec ${SIF_NAME} python -c \"import platform; print(platform.machine())\"   # expect x86_64"
echo "  apptainer exec ${SIF_NAME} python -c \"import pymbar, numpy; print('OK', pymbar.__version__, numpy.__version__)\""

