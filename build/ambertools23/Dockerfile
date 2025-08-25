# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bookworm

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/ambertools23 \
    PATH=/opt/conda/envs/ambertools23/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# still working on this: jupyter_client=10.6.1 jupyter_core=5.7.2 && \
# --- conda env with pinned core stack (CONDA syntax: = not ==) ---
RUN micromamba create -y -n ambertools23 -c conda-forge python=3.11 && \
    micromamba install -y -n ambertools23 -c conda-forge ambertools=23 

# Install the patched module robustly:
#  - compute the actual site-packages path
#  - ensure the pymbar dir exists
#  - copy the file
#  - slim the image
RUN PYSP=$(/opt/conda/envs/ambertools23/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    micromamba clean -a -y && \
    find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /opt/conda/envs/ambertools23 -type d \( -name tests -o -name test -o -name doc -o -name docs -o -name examples \) -prune -exec rm -rf {} + && \
    find /opt/conda/envs/ambertools23 -type f \( -name "*.a" -o -name "*.la" \) -delete

