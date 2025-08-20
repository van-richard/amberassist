# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bookworm

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/mbar \
    PATH=/opt/conda/envs/mbar/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Create env + conda-forge pins
# still working on this: jupyter_client=10.6.1 jupyter_core=5.7.2 && \
RUN micromamba create -y -n mbar -c conda-forge python=3.9.19 && \
    micromamba install -y -n mbar -c conda-forge \
      pip=23.3.1 \
    /opt/conda/envs/mbar/bin/pip install --no-cache-dir \
      pymbar==3.0.5 scikit-learn==1.4.2 numpy==1.26.4 scipy=1.13.0 pandas=2.2.1 \
      matplotlib=3.8.4 seaborn-base=0.13.2 ipython=8.18.1 ipykernel=6.29.3 \

# Fetch your exact patched file (locked to commit e15e7b8...)
# (You can replace this ADD with a local COPY if you prefer to avoid network)
# Replace your ADD line with this:
ADD --chmod=0644 https://raw.githubusercontent.com/van-richard/amberassist/e15e7b873fe5548783f76114cce4d58292254f69/mbar/mbar_pmf.py /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/mbar_pmf.py 

# Install the patched module robustly:
#  - compute the actual site-packages path
#  - ensure the pymbar dir exists
#  - copy the file
#  - slim the image
RUN PYSP=$(/opt/conda/envs/mbar/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    install -d "${PYSP}/pymbar" && \
    micromamba clean -a -y && \
    find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find /opt/conda/envs/mbar -type d \( -name tests -o -name test -o -name doc -o -name docs -o -name examples \) -prune -exec rm -rf {} + && \
    find /opt/conda/envs/mbar -type f \( -name "*.a" -o -name "*.la" \) -delete

# Executable behavior like a SIF
ENTRYPOINT ["/opt/conda/envs/mbar/bin/python"]

