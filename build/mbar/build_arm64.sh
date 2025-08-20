#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
IMG_TAG=${IMG_TAG:-mbar:arm64}
TAR_NAME=${TAR_NAME:-mbar_arm64.tar}
SIF_NAME=${SIF_NAME:-mbar_arm64.sif}           # local SIF name for your Mac/Lima
LIMA_INSTANCE=${LIMA_INSTANCE:-apptainer}
REMOTE_DIR=/tmp
REMOTE_TAR="${REMOTE_DIR}/${TAR_NAME}"
REMOTE_SIF="${REMOTE_DIR}/${SIF_NAME}"

# ---------- checks ----------
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v limactl >/dev/null || { echo "limactl not found"; exit 1; }
docker buildx ls >/dev/null || { echo "Docker Buildx not available"; exit 1; }

echo "==> Step 1: Build linux/arm64 image and export docker-archive tar"
docker buildx build --platform linux/arm64 -t "${IMG_TAG}" .
docker save "${IMG_TAG}" -o "${TAR_NAME}"
echo "   created ${TAR_NAME}"

echo "==> Step 2: Copy tar into Lima VM (${LIMA_INSTANCE}) at ${REMOTE_TAR}"
limactl copy "${TAR_NAME}" "${LIMA_INSTANCE}:${REMOTE_TAR}"

echo "==> Step 3: Convert tar -> SIF inside Lima (force arm64), non-interactive"
limactl shell "${LIMA_INSTANCE}" -- bash -lc "
  set -euo pipefail
  cd '${REMOTE_DIR}'
  export APPTAINER_DOCKER_ARCH=arm64
  export SINGULARITY_DOCKER_ARCH=arm64
  apptainer build --force --arch arm64 '${SIF_NAME}' docker-archive://'${TAR_NAME}'
  echo '   built ${SIF_NAME} in ${REMOTE_DIR}'
"

echo "==> Step 4: Copy SIF back to macOS"
limactl copy "${LIMA_INSTANCE}:${REMOTE_SIF}" "./${SIF_NAME}"

echo "==> Done."
echo "SIF: $(pwd)/${SIF_NAME}"
echo
echo "Run inside Lima (arm64):"
echo "  limactl shell ${LIMA_INSTANCE} -- apptainer exec ${REMOTE_SIF} python -c \"import platform; print(platform.machine())\"  # expect aarch64/arm64"

