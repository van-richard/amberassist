#!/bin/bash
# Prepare topology/coordinate files with normal 12-6 Lennard-Jones parameters.
# For protein, nucleic acid, and ion-containing systems.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") system.pdb

Environment overrides:
  OUT_PREFIX   Output prefix (default: step3_pbcsetup)
  WATER_BOX    Water box model (default: TIP3PBOX)
  BUFFER       Solvent buffer distance (default: 12.0)
  CLOSENESS    Solvent closeness parameter (default: 0.8)
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
WATER_BOX=${WATER_BOX:-TIP3PBOX}
BUFFER=${BUFFER:-12.0}
CLOSENESS=${CLOSENESS:-0.8}

if [ ! -f "$SYS" ]; then
    echo "ERROR: Input PDB file not found: $SYS" >&2
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

sys = loadpdb ${SYS}

savepdb sys ${OUT_PREFIX}.pdb
charge sys
solvatebox sys ${WATER_BOX} ${BUFFER} iso ${CLOSENESS}
addions sys Na+ 0
addions sys Cl- 0

saveamberparm sys ${OUT_PREFIX}.parm7 ${OUT_PREFIX}.rst7
savepdb sys ${OUT_PREFIX}_wat.pdb

quit
_EOF
