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

<<<<<<< HEAD:build/ambertools23/Dockerfile
# still working on this: jupyter_client=10.6.1 jupyter_core=5.7.2 && \
# --- conda env with pinned core stack (CONDA syntax: = not ==) ---
RUN micromamba create -y -n ambertools23 -c conda-forge python=3.11 && \
    micromamba install -y -n ambertools23 -c conda-forge ambertools=23 
=======
# Create AmberTools 24 env (minimal change from v23 → v24)
RUN micromamba create -y -n ambertools24 -c conda-forge \
      python=3.11 pip ambertools=24 \
 && micromamba clean -a -y
>>>>>>> test:build/ambertools/Dockerfile

# Standard Amber env var (same path convention as before)
ENV AMBERHOME=/opt/conda/envs/ambertools24

# Non-root user provided by base image
WORKDIR /work
USER $MAMBA_USER

CMD ["bash"]
