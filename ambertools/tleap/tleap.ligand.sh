#!/bin/bash
# Build standalone topology/coordinate files for a custom ligand.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") ligand[.mol2]

Environment overrides:
  OUT_PREFIX   Output prefix (default: ligand stem)
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -ne 1 ]; then
    echo "ERROR: Requires exactly one ligand MOL2 file or ligand name." >&2
    usage >&2
    exit 1
fi

ligand=$1
case "$ligand" in
    *.mol2)
        mol2_file=$ligand
        ;;
    *.*)
        stem=${ligand%.*}
        mol2_file="${stem}.mol2"
        ;;
    *)
        mol2_file="${ligand}.mol2"
        ;;
esac

stem=${mol2_file%.mol2}
frcmod_file="${stem}.frcmod"
OUT_PREFIX=${OUT_PREFIX:-$stem}

if [ ! -f "$mol2_file" ]; then
    echo "ERROR: Ligand MOL2 file not found: $mol2_file" >&2
    exit 1
fi

if [ ! -f "$frcmod_file" ]; then
    echo "ERROR: Ligand frcmod file not found: $frcmod_file" >&2
    echo "Run parmchk2 before this script." >&2
    exit 1
fi

if ! command -v tleap >/dev/null 2>&1; then
    echo "ERROR: tleap was not found. Load AmberTools before running this script." >&2
    exit 1
fi

tleap -f - <<_EOF
source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.gaff2

loadamberparams ${frcmod_file}
LIG = loadmol2 ${mol2_file}

check LIG
saveoff LIG ${OUT_PREFIX}.lib
saveamberparm LIG ${OUT_PREFIX}.parm7 ${OUT_PREFIX}.rst7

quit
_EOF
