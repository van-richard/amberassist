# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bookworm

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/miniforge3 \
    PATH=/opt/conda/envs/miniforge3/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# still working on this: jupyter_client=10.6.1 jupyter_core=5.7.2 && \
# --- conda env with pinned core stack (CONDA syntax: = not ==) ---
RUN micromamba create -y -n miniforge3 -c conda-forge python=3.9.19 && \
    micromamba install -y -n miniforge3 -c conda-forge \
      pip=23.3.1 ipython=8.18.1 ipykernel=6.29.3 \
      matplotlib=3.8.4 seaborn-base=0.13.2 \
      numpy=1.26.4 scipy=1.13.0 pandas=2.2.1 \
      jupyterlab \
      libblas=3.9.0=22_linux64_openblas libcblas=3.9.0=22_linux64_openblas \
      liblapack=3.9.0=22_linux64_openblas libopenblas=0.3.27 \
      tzdata=2024a ca-certificates

# Install the patched module robustly:
#  - compute the actual site-packages path
#  - ensure the pyminiforge3 dir exists
#  - copy the file
#  - slim the image
RUN PYSP=$(/opt/conda/envs/miniforge3/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    install -d "${PYSP}/pyminiforge3" && \
    micromamba clean -a -y && \
    find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /opt/conda/envs/miniforge3 -type d \( -name tests -o -name test -o -name doc -o -name docs -o -name examples \) -prune -exec rm -rf {} + && \
    find /opt/conda/envs/miniforge3 -type f \( -name "*.a" -o -name "*.la" \) -delete
