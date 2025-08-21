#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# -------- Config (override via env) --------
IMG_TAG="${IMG_TAG:-miniforge3:amd64}"
TAR_NAME="${TAR_NAME:-miniforge3_amd64.tar}"
SIF_NAME="${SIF_NAME:-miniforge3_amd64.sif}"
LIMA_INSTANCE="${LIMA_INSTANCE:-apptainer}"  # lima instance name (limactl ls)
REMOTE_DIR="/tmp"
REMOTE_TAR="${REMOTE_DIR}/${TAR_NAME}"
REMOTE_SIF="${REMOTE_DIR}/${SIF_NAME}"

# -------- Helpers --------
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

run_lima() {
# Run non-interactively inside Lima VM
limactl shell "${LIMA_INSTANCE}" -- bash -lc "$*"
}

copy_to_lima()   { limactl copy "$1" "${LIMA_INSTANCE}:$2"; }
copy_from_lima() { limactl copy "${LIMA_INSTANCE}:$1" "$2"; }

# -------- Sanity checks --------
need docker
need limactl
docker buildx ls >/dev/null 2>&1 || die "Docker Buildx not available (enable Docker Desktop Buildx or Colima buildx)."
limactl ls | grep -q "^${LIMA_INSTANCE}\b" || die "Lima instance '${LIMA_INSTANCE}' not found. Run: limactl ls"

#copy_to_lima "../../miniforge3/miniforge3_pmf.py" "/tmp/miniforge3_pmf.py"
#copy_to_lima "../../miniforge3/init.PATCH" "/tmp/init.PATCH"

# Ensure Dockerfile exists
[ -f Dockerfile ] || die "Dockerfile not found in $(pwd)"

echo "==> 1/4 Build linux/amd64 image (QEMU) and export docker-archive tar"
docker buildx build --platform linux/amd64 -t "${IMG_TAG}" .
docker save "${IMG_TAG}" -o "${TAR_NAME}"
[ -s "${TAR_NAME}" ] || die "Tar not created: ${TAR_NAME}"
echo "    created ${TAR_NAME}"

echo "==> 2/4 Copy tar into Lima VM (${LIMA_INSTANCE}) at ${REMOTE_TAR}"
copy_to_lima "${TAR_NAME}" "${REMOTE_TAR}"

echo "==> 3/4 Convert tar -> SIF inside Lima (force amd64), non-interactive"
run_lima "
set -Eeuo pipefail
command -v apptainer >/dev/null || { echo 'apptainer not found in VM'; exit 1; }
cd '${REMOTE_DIR}'
export APPTAINER_DOCKER_ARCH=amd64
export SINGULARITY_DOCKER_ARCH=amd64
apptainer build --force --arch amd64 '${SIF_NAME}' docker-archive://'${TAR_NAME}'
[ -s '${SIF_NAME}' ] || { echo 'SIF not created'; exit 1; }
echo '    built ${SIF_NAME} in ${REMOTE_DIR}'
"

echo "==> 4/4 Copy SIF back to macOS"
copy_from_lima "${REMOTE_SIF}" "./${SIF_NAME}"
[ -s "${SIF_NAME}" ] || die "Failed to copy SIF back from Lima"

echo
echo "Done. SIF: $(pwd)/${SIF_NAME}"
echo "Quick tests (if apptainer is installed on macOS):"
echo "  apptainer exec ${SIF_NAME} python -c 'import platform; print(platform.machine())'   # expect x86_64"
echo "  apptainer exec ${SIF_NAME} python -c 'import pyminiforge3, numpy; print(\"OK\", pyminiforge3.__version__, numpy.__version__)'"

