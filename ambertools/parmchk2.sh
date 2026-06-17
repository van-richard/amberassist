#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") ligand[.mol2]

Generate an frcmod file from a ligand MOL2 file with AmberTools parmchk2.

The ligand argument can be either a MOL2 file or a ligand name. If a name is
provided, this script looks for <name>.mol2.

Examples:
  $(basename "$0") ATP.mol2
  $(basename "$0") ATP
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

if [ ! -e "$mol2_file" ]; then
    echo "ERROR: MOL2 file not found: $mol2_file" >&2
    exit 1
fi

if [ ! -f "$mol2_file" ]; then
    echo "ERROR: MOL2 path is not a regular file: $mol2_file" >&2
    exit 1
fi

if ! command -v parmchk2 >/dev/null 2>&1; then
    echo "ERROR: parmchk2 was not found. Load AmberTools before running this script." >&2
    exit 1
fi

frcmod_file="${mol2_file%.mol2}.frcmod"

if [ -f "$frcmod_file" ]; then
    echo "Found frcmod file: $frcmod_file"
    echo "Skipping parmchk2."
    exit 0
fi

parmchk2 -i "$mol2_file" -f mol2 -o "$frcmod_file"

echo "Created $frcmod_file"
