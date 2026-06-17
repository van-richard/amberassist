#!/bin/bash
# Combine one custom ligand with one protein, then solvate and neutralize.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") ligand[.mol2]

Environment overrides:
  PROTEIN      Protein PDB file (default: protein_clean.pdb)
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
lib_file="${stem}.lib"
PROTEIN=${PROTEIN:-protein_clean.pdb}
OUT_PREFIX=${OUT_PREFIX:-step3_pbcsetup}
WATER_BOX=${WATER_BOX:-TIP3PBOX}
BUFFER=${BUFFER:-12.0}
CLOSENESS=${CLOSENESS:-0.8}

for required_file in "$mol2_file" "$frcmod_file" "$lib_file" "$PROTEIN"; do
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

lig = loadmol2 ${mol2_file}
pro = loadpdb ${PROTEIN}
sys = combine { lig pro }

solvatebox sys ${WATER_BOX} ${BUFFER} iso ${CLOSENESS}
addions sys Na+ 0
addions sys Cl- 0

savepdb sys ${OUT_PREFIX}_wat.pdb
saveamberparm sys ${OUT_PREFIX}.parm7 ${OUT_PREFIX}.rst7

quit
_EOF
