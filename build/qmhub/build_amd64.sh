# syntax=docker/dockerfile:1.6
FROM --platform=linux/amd64 condaforge/miniforge3:24.5.0-0

SHELL ["/bin/bash", "-lc"]
ARG ENV_NAME=qmhub
ARG PY_VER=3.10

# OS deps: compiler stack, CMake, FFTW for helPME
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        libfftw3-dev git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create runtime env (qmhub requires python>=3.8, numpy, scipy)
ENV DEBIAN_FRONTEND=noninteractive
RUN conda update -y -n base conda && \
    conda create -y -n ${ENV_NAME} python=${PY_VER} numpy scipy pip && \
    conda clean -afy

# Make conda env the default
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${ENV_NAME}

# Avoid oversubscription by default (override in SLURM scripts as needed)
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Bring sources into the image
WORKDIR /opt/src
# Expect the host to provide ./qmhub and ./helpme in the build context:
COPY qmhub /opt/src/qmhub
COPY helpme /opt/src/helpme

# 1) Install qmhub into the env
RUN source activate ${ENV_NAME} && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir /opt/src/qmhub

# 2) Build + install helPME's Python module (helpmelib) with CMake
#    - Build type Release, OpenMP ON, MPI OFF for portability
#    - Use env python for bindings
RUN source activate ${ENV_NAME} && \
    cd /opt/src/helpme && mkdir -p build && cd build && \
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_OPENMP=ON \
      -DENABLE_MPI=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/helpme \
      -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" && \
    cmake --build . -j"$(nproc)" && \
    # Install the Python module into the active conda env via CMake helper target
    cmake --build . --target PythonInstall && \
    # Verify top-level import works
    python -c 'import helpmelib; print(\"helpmelib import OK\")' && \
    # Copy the compiled extension into the qmhub package namespace
    HELPSO="$(find . -name 'helpmelib*.so' -type f | head -n1)" && \
    PY_SITE="$(python - <<'PY'\nimport site,sys\nc=[*site.getsitepackages(), site.getusersitepackages()]\nprint([p for p in c if 'site-packages' in p][0])\nPY\n)" && \
    install -d "${PY_SITE}/qmhub" && \
    install -m 0644 "${HELPSO}" "${PY_SITE}/qmhub/" && \
    # Sanity check: qmhub.helpmelib must import and expose PME types
    python - <<'PY'\nimport importlib; m=importlib.import_module('qmhub.helpmelib'); print('qmhub.helpmelib OK', hasattr(m,'PMEInstanceD'))\nPY

# Final image defaults
WORKDIR /work
ENTRYPOINT ["/bin/bash"]

