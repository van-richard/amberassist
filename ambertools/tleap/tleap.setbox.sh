#!/bin/bash
# Build a system from explicit components and assign a custom box size.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0")

Environment overrides:
  PROTEIN      Protein PDB file (default: protein.pdb)
  LIGAND       Ligand stem or MOL2 file (default: lig)
  ION          Ion PDB file (default: ion.pdb)
  WATER        Water PDB file (default: water.pdb)
  BOX_SIZE     Box dimensions (default: 82.0 82.0 82.0)
  OUT_PREFIX   Output prefix (default: step3_pbcsetup)
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -ne 0 ]; then
    echo "ERROR: This script uses environment variables for inputs; pass no positional arguments." >&2
    usage >&2
    exit 1
fi

PROTEIN=${PROTEIN:-protein.pdb}
LIGAND=${LIGAND:-lig}
ION=${ION:-ion.pdb}
WATER=${WATER:-water.pdb}
BOX_SIZE=${BOX_SIZE:-82.0 82.0 82.0}
OUT_PREFIX=${OUT_PREFIX:-step3_pbcsetup}

case "$LIGAND" in
    *.mol2)
        mol2_file=$LIGAND
        ;;
    *.*)
        stem=${LIGAND%.*}
        mol2_file="${stem}.mol2"
        ;;
    *)
        mol2_file="${LIGAND}.mol2"
        ;;
esac

stem=${mol2_file%.mol2}
frcmod_file="${stem}.frcmod"
lib_file="${stem}.lib"

for required_file in "$PROTEIN" "$ION" "$WATER" "$mol2_file" "$frcmod_file" "$lib_file"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required file not found: $required_file" >&2
        exit 1
    fi
done

if ! command -v tleap >/dev/null 2>&1; then
    echo "ERROR: tleap was not found. Load AmberTools before running this script." >&2
    exit 1
fi

tleap -f - <<_EOF
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.gaff2

loadamberparams ${frcmod_file}
loadoff ${lib_file}

PROTEIN = loadpdb ${PROTEIN}
LIG = loadmol2 ${mol2_file}
ION = loadpdb ${ION}
WAT = loadpdb ${WATER}

SYS = combine { PROTEIN LIG ION WAT }

set SYS box { ${BOX_SIZE} }
check SYS

savepdb SYS ${OUT_PREFIX}.pdb
saveamberparm SYS ${OUT_PREFIX}.parm7 ${OUT_PREFIX}.rst7

quit
_EOF
