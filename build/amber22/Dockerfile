# Multi-stage Dockerfile to (1) build toolchain + AmberTools sander/sander.MPI,
# (2) build helpmelib from helpme.zip, (3) install qmhub from qmhub.zip,
# then (4) assemble a lean runtime. Targeted for linux/amd64 runtime.

# ---- Base build image --------------------------------------------------------
FROM ubuntu:22.04 AS base-build
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Common build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget git cmake make ninja-build build-essential \
      gfortran g++ gcc gdb valgrind \
      python3 python3-dev python3-pip python3-venv \
      csh tcsh patch bison flex file \
      libx11-dev libxext-dev libxft-dev libxrender-dev \
      liblapack-dev libblas-dev \
      zlib1g-dev libbz2-dev libzstd-dev \
      libnetcdf-dev netcdf-bin \
      mpich libmpich-dev \
    && rm -rf /var/lib/apt/lists/*

# Use a venv for python packages used at build time
RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate && \
    python -m pip install --upgrade pip wheel setuptools

# Staging of your provided archives
WORKDIR /opt/src
# Place these two files next to the Dockerfile before building
COPY qmhub.zip /opt/src/qmhub.zip
COPY helpme.zip /opt/src/helpme.zip

# Optional: AmberTools tarball (offline-friendly).
# If you have a local AmberTools22 source tarball, drop it as:
#   ambertools22.tar.bz2
# Otherwise we'll try to fetch it in the next step.
COPY ambertools22.tar.bz2 /opt/src/ambertools22.tar.bz2 2>/dev/null || true

# ---- Build AmberTools (sander + sander.MPI) ---------------------------------
FROM base-build AS amber-build
ENV AMBER_PREFIX=/opt/amber22 \
    AMBERHOME=/opt/amber22 \
    AMBERTOOLS_TARBALL=/opt/src/ambertools22.tar.bz2 \
    NPROC=8

WORKDIR /opt/build/amber

# Fetch AmberTools22 tarball if not provided offline.
# Prefer offline tarball; else attempt download (multiple mirrors tried).
RUN set -euxo pipefail; \
    if [[ ! -s "${AMBERTOOLS_TARBALL}" ]]; then \
      echo "AmberTools22 tarball not found locally. Attempting download..."; \
      urls=( \
        "https://ambermd.org/downloads/AmberTools22.tar.bz2" \
        "https://ftp.rcsb.org/pub/software/amber/AmberTools22.tar.bz2" \
      ); \
      for u in "${urls[@]}"; do \
        if curl -fsSL --retry 3 "${u}" -o /opt/src/ambertools22.tar.bz2; then \
          echo "Downloaded AmberTools22 from ${u}"; break; \
        fi; \
      done; \
    fi

# Unpack
RUN tar -xf /opt/src/ambertools22.tar.bz2 --strip-components=0 -C /opt/build/amber

# Apply user patches if present:
# Expect your patch files under patches/ amber22_src/, sinr/, sqm/, asm/, qmhub/ (order matters)
# Each .patch file will be applied with -p0 when a matching path exists.
COPY patches/ /opt/patches/
RUN set -euo pipefail; \
    shopt -s nullglob; \
    pushd /opt/build/amber >/dev/null; \
    for d in amber22_src qmhub sinr sqm asm; do \
      if [[ -d "/opt/patches/${d}" ]]; then \
        for p in /opt/patches/${d}/*.patch; do \
          echo "Applying patch ${p} ..."; \
          patch -p0 < "${p}"; \
        done; \
      fi; \
    done; \
    popd >/dev/null

# Configure & build:
#  - We only need sander / sander.MPI; use MPICH to maximize ABI compatibility with Intel MPI on HPC.
#  - Disable X and OpenMP to simplify runtime.
#  - AmberTools 22 uses the legacy configure—this works well in containers.
ENV PATH="${AMBERHOME}/bin:/opt/venv/bin:${PATH}"

RUN set -euxo pipefail; \
    cd /opt/build/amber; \
    # Some Amber scripts insist on AMBERHOME existing first:
    mkdir -p "${AMBERHOME}"; \
    # Configure serial (for sander) and MPI (for sander.MPI)
    ./configure -noX -noOpenMP gnu; \
    ./configure -noX -noOpenMP -mpi gnu; \
    make -j"${NPROC}" install; \
    # Quick sanity: ensure sander and sander.MPI exist
    test -x "${AMBERHOME}/bin/sander"; \
    test -x "${AMBERHOME}/bin/sander.MPI"

# ---- Build helpmelib ---------------------------------------------------------
FROM base-build AS helpme-build
WORKDIR /opt/build/helpme
COPY --from=base-build /opt/src/helpme.zip /opt/build/helpme/helpme.zip
RUN unzip -q helpme.zip && rm helpme.zip && \
    # Build in Release
    cmake -S helpme -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    cmake --build build -j && \
    cmake --install build --prefix /opt/helpmelib

# ---- Install qmhub (with Q-Chem failure guard) -------------------------------
FROM base-build AS qmhub-build
# Bring in helpmelib built artifacts
COPY --from=helpme-build /opt/helpmelib /opt/helpmelib
ENV PATH=/opt/helpmelib/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/helpmelib/lib:${LD_LIBRARY_PATH} \
    HELPMELIB_PREFIX=/opt/helpmelib

# Unpack & install qmhub from provided archive
WORKDIR /opt/src
COPY --from=base-build /opt/src/qmhub.zip /opt/src/qmhub.zip
RUN unzip -q qmhub.zip && rm qmhub.zip

# Patch-in a Q-Chem failure guard without touching upstream logic:
# We add a tiny wrapper and encourage users to point QChem to it via QCHEM env var.
# Wrapper will detect non-zero exit, or common failure strings, and will try to stop MPI/SLURM cleanly.
RUN mkdir -p /opt/qmhub-extras/bin
COPY scripts/qchem_wrapper.sh /opt/qmhub-extras/bin/qchem_wrapper.sh
RUN chmod +x /opt/qmhub-extras/bin/qchem_wrapper.sh

# Install qmhub into the venv (wheel preferred)
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir ./qmhub

# ---- Final runtime image -----------------------------------------------------
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Minimal runtime deps (python, mpi)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-distutils python3-venv python3-minimal \
      mpich libmpich12 libgfortran5 \
      ca-certificates zlib1g libbz2-1.0 libzstd1 liblapack3 libblas3 \
    && rm -rf /var/lib/apt/lists/*

# Amber (sander + sander.MPI)
COPY --from=amber-build /opt/amber22 /opt/amber22
ENV AMBERHOME=/opt/amber22 \
    PATH=/opt/amber22/bin:${PATH}

# Python venv and qmhub installed in the build stage
COPY --from=base-build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# helpmelib runtime
COPY --from=helpme-build /opt/helpmelib /opt/helpmelib
ENV PATH=/opt/helpmelib/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/helpmelib/lib:${LD_LIBRARY_PATH} \
    HELPMELIB_PREFIX=/opt/helpmelib

# Extra scripts (Q-Chem guard, env setup)
COPY --from=qmhub-build /opt/qmhub-extras /opt/qmhub-extras
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Basic self-test script inside the image
COPY scripts/apptest.sh /usr/local/bin/apptest.sh
RUN chmod +x /usr/local/bin/apptest.sh

# Default environment
ENV OMP_NUM_THREADS=1 \
    QCHEM_WRAPPER=/opt/qmhub-extras/bin/qchem_wrapper.sh

# Default command prints versions and a quick check
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["apptest"]

