# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bookworm

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/qmhub \
    PATH=/opt/conda/envs/qmhub/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

SHELL ["/bin/bash", "-lc"]
ARG ENV_NAME=qmhub
ARG PY_VER=3.10

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Create a minimal env for qmhub runtime deps (from pyproject: numpy, scipy)
RUN micromamba create -y -n ${ENV_NAME} -c conda-forge python=${PY_VER} && \
    micromamba install -y -n ${ENV_NAME} -c conda-forge numpy scipy pip && \
    micromamba clean -afy

# Copy the source tree (expecting ./qmhub in build context)
WORKDIR /opt/src
COPY qmhub /opt/src/qmhub

# Install qmhub into the conda env and sanity check
RUN source activate ${ENV_NAME} && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir /opt/src/qmhub && \
    python - <<'PY'
import qmhub, numpy, scipy
print("OK", qmhub.__version__, numpy.__version__, scipy.__version__)
PY

# Make the env default
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${ENV_NAME}

# Default workdir and entry
WORKDIR /work
ENTRYPOINT ["/bin/bash"]

