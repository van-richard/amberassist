#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# -------- Config (override via env) --------
IMG_TAG="${IMG_TAG:-mbar:amd64}"
TAR_NAME="${TAR_NAME:-mbar_amd64.tar}"
SIF_NAME="${SIF_NAME:-mbar_amd64.sif}"
REMOTE_DIR="/tmp"
REMOTE_TAR="${REMOTE_DIR}/${TAR_NAME}"
REMOTE_SIF="${REMOTE_DIR}/${SIF_NAME}"

# -------- Helpers --------
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

# -------- Sanity checks --------
cp ../../mbar/mbar_pmf.py .
cp ../../mbar/init.PATCH .
need docker
docker buildx ls >/dev/null 2>&1 || die "Docker Buildx not available (enable Docker Desktop Buildx or Colima buildx)."

# Ensure Dockerfile exists
[ -f Dockerfile ] || die "Dockerfile not found in $(pwd)"

echo "==> 1/4 Build linux/amd64 image (QEMU) and export docker-archive tar"
docker buildx build --platform linux/amd64 -t "${IMG_TAG}" .
docker save "${IMG_TAG}" -o "${TAR_NAME}"
[ -s "${TAR_NAME}" ] || die "Tar not created: ${TAR_NAME}"
echo "    created ${TAR_NAME}"

echo "==> 3/4 Convert tar -> SIF (force amd64), non-interactive"
export APPTAINER_DOCKER_ARCH=amd64
export SINGULARITY_DOCKER_ARCH=amd64
apptainer build --force --arch amd64 "${SIF_NAME}" docker-archive://"${TAR_NAME}"
[ -s "${SIF_NAME}" ] || { echo 'SIF not created'; exit 1; }
echo "    built ${SIF_NAME} in ${REMOTE_DIR}"

echo "==> 4/4 Copy SIF back to linux"
[ -s "${SIF_NAME}" ] || die "Failed to copy SIF back from Lima"

echo
echo "Done. SIF: $(pwd)/${SIF_NAME}"
echo "Quick tests (if apptainer is installed on macOS):"
echo "  apptainer exec ${SIF_NAME} python -c 'import platform; print(platform.machine())'   # expect x86_64"
echo "  apptainer exec ${SIF_NAME} python -c 'import pymbar, numpy; print(\"OK\", pymbar.__version__, numpy.__version__)'"

rm init.PATCH mbar_pmf.py
