# syntax=docker/dockerfile:1
# AmberTools 24 container with micromamba, pinned base for stability.
FROM mambaorg/micromamba:1.5.8-bookworm

SHELL ["/bin/bash", "-lc"]

# Match prior environment layout; only bump env name/version.
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/ambertools24 \
    PATH=/opt/conda/envs/ambertools24/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Create AmberTools 24 env.
RUN micromamba create -y -n ambertools24 -c conda-forge \
      python=3.11 pip ambertools=24 \
 && micromamba clean -a -y

# Standard Amber env var (same path convention as before)
ENV AMBERHOME=/opt/conda/envs/ambertools24

# Non-root user provided by base image
WORKDIR /work
USER $MAMBA_USER

CMD ["bash"]
