#!/usr/bin/env bash
set -euo pipefail

# Build an AMD64 (x86_64) image for Linux on an Apple Silicon (arm64) host.
# Requires Docker Buildx and binfmt/qemu for cross-builds.

IMAGE_TAG="${IMAGE_TAG:-ambertools:24-amd64}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CONTEXT="${CONTEXT:-.}"
BUILDER_NAME="${BUILDER_NAME:-apptainer}"

# Ensure a buildx builder exists and is selected
if ! docker buildx ls | grep -q "${BUILDER_NAME}"; then
	docker buildx create --name "${BUILDER_NAME}" --use >/dev/null
else
	docker buildx use "${BUILDER_NAME}" >/dev/null
fi

# Ensure binfmt is installed for cross-arch emulation
docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || true

# Bootstrap the builder
docker buildx inspect --bootstrap >/dev/null

# Build and load into the local Docker engine
docker buildx build \
	--platform linux/amd64 \
	-t "${IMAGE_TAG}" \
	-f "${DOCKERFILE}" \
	"${CONTEXT}" \
	--load

echo "Built and loaded ${IMAGE_TAG} (linux/amd64)."

