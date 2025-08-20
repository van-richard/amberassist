# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8-bookworm

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    CONDA_PREFIX=/opt/conda/envs/mbar \
    PATH=/opt/conda/envs/mbar/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# still working on this: jupyter_client=10.6.1 jupyter_core=5.7.2 && \
# --- conda env with pinned core stack (CONDA syntax: = not ==) ---
RUN micromamba create -y -n mbar -c conda-forge python=3.9.19 && \
    micromamba install -y -n mbar -c conda-forge \
      pip=23.3.1 setuptools=68.2.2 wheel=0.41.2 \
      ipython=8.18.1 ipykernel=6.29.3 \
      matplotlib=3.8.4 seaborn-base=0.13.2 \
      numpy=1.26.4 scipy=1.13.0 pandas=2.2.1 \
      libblas=3.9.0=22_linux64_openblas libcblas=3.9.0=22_linux64_openblas \
      liblapack=3.9.0=22_linux64_openblas libopenblas=0.3.27 \
      tzdata=2024a ca-certificates

# --- pip packages (PIP syntax: ==) ---
RUN /opt/conda/envs/mbar/bin/pip install --no-cache-dir \
      pymbar==3.0.5 scikit-learn==1.4.2 joblib==1.4.0 threadpoolctl==3.4.0

RUN mv /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/__init__.py /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/init.BAK


# --- stage your patched file (networked) ---
#ADD --chmod=0644 https://github.com/van-richard/amberassist/blob/bed42c902549bff2e8fe6283809f029950db1d23/mbar/mbar_pmf.py /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/mbar_pmf.py
#ADD --chmod=0644 https://github.com/van-richard/amberassist/blob/9c88882cff0a8859e80939fbf70824ee3bb7e76b/mbar/init.PATCH /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/__init__.py


# Fetch your exact patched file (locked to commit e15e7b8...)
# (You can replace this ADD with a local COPY if you prefer to avoid network)
# Replace your ADD line with this:
#COPY --chmod=0644 mbar_pmf.py /tmp/mbar_pmf.py
# /opt/conda/envs/mbar/lib/python3.9/site-packages/pymbar/mbar_pmf.py 
# ADD --chmod=0644  https://github.com/van-richard/amberassist/blob/bed42c902549bff2e8fe6283809f029950db1d23/mbar/mbar_pmf.py 

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

COPY --chmod=0644 mbar_pmf.py /tmp/mbar_pmf.py
COPY --chmod=0644 init.PATCH /tmp/__init__.py
RUN PYSP=$(/opt/conda/envs/mbar/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    install -D -m 0644 /tmp/mbar_pmf.py "${PYSP}/pymbar/mbar_pmf.py" && \
    install -D -m 0644 /tmp/__init__.py "${PYSP}/pymbar/__init__.py"

