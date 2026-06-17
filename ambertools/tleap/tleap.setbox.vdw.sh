#!/bin/bash
# Prepare topology/coordinate files with a box based on van der Waals radii.
# Uses 12-6-4 Lennard-Jones ion parameters by default.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") system.pdb

Environment overrides:
  OUT_PREFIX   Output prefix (default: step3_pbcsetup)
  ION_FRCMOD   12-6-4 ion frcmod file (default: frcmod.ions234lm_1264_tip3p)
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "ERROR: Requires exactly one input PDB file." >&2
    usage >&2
    exit 1
fi

SYS=$1
OUT_PREFIX=${OUT_PREFIX:-step3_pbcsetup}
ION_FRCMOD=${ION_FRCMOD:-frcmod.ions234lm_1264_tip3p}

if [ ! -f "$SYS" ]; then
    echo "ERROR: Input PDB file not found: $SYS" >&2
    exit 1
fi

if [ ! -f "$ION_FRCMOD" ]; then
    echo "ERROR: 12-6-4 ion frcmod file not found: $ION_FRCMOD" >&2
    exit 1
fi

if ! command -v tleap >/dev/null 2>&1; then
    echo "ERROR: tleap was not found. Load AmberTools before running this script." >&2
    exit 1
fi

tleap -f - <<_EOF
source leaprc.protein.ff14SB
source leaprc.DNA.OL15
source leaprc.RNA.OL3
source leaprc.water.tip3p
loadamberparams ${ION_FRCMOD}

sys = loadpdb ${SYS}

savepdb sys ${OUT_PREFIX}.pdb
charge sys
setbox sys vdw

saveamberparm sys ${OUT_PREFIX}.parm7 ${OUT_PREFIX}.rst7
savepdb sys ${OUT_PREFIX}_wat.pdb

quit
_EOF
