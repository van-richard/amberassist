#!/bin/bash

set -euo pipefail

charge_method="bcc"
verbose=2
net_charge=0
resname="UNL"
intermediate_files="yes"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-r RESNAME] [-n NET_CHARGE] ligand.pdb

Generate a GAFF2 MOL2 file from a ligand PDB file with AmberTools antechamber.

Options:
  -r RESNAME      Residue name to write to the MOL2 file (default: UNL)
  -n NET_CHARGE   Net molecular charge (default: 0)
  -h              Show this help message

Examples:
  $(basename "$0") ATP.pdb
  $(basename "$0") -r ATP -n -4 ATP.pdb
EOF
}

if [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

while getopts ":r:n:h" opt; do
    case "$opt" in
        r)
            resname="$OPTARG"
            ;;
        n)
            net_charge="$OPTARG"
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "ERROR: Option -$OPTARG requires an argument." >&2
            usage >&2
            exit 1
            ;;
        \?)
            echo "ERROR: Unknown option -$OPTARG." >&2
            usage >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [ "$#" -ne 1 ]; then
    echo "ERROR: Requires exactly one ligand PDB file." >&2
    usage >&2
    exit 1
fi

ligand=$1

case "$ligand" in
    *.pdb)
        ;;
    *)
        echo "ERROR: antechamber.sh only accepts PDB input files (*.pdb): $ligand" >&2
        exit 1
        ;;
esac

if [ ! -e "$ligand" ]; then
    echo "ERROR: Ligand PDB file not found: $ligand" >&2
    exit 1
fi

if [ ! -f "$ligand" ]; then
    echo "ERROR: Ligand path is not a regular file: $ligand" >&2
    exit 1
fi

if ! command -v antechamber >/dev/null 2>&1; then
    echo "ERROR: antechamber was not found. Load AmberTools before running this script." >&2
    exit 1
fi

filename=$(basename "$ligand")
stem=${filename%.pdb}
output="${stem}.mol2"

antechamber \
    -i "$ligand" \
    -fi pdb \
    -o "$output" \
    -fo mol2 \
    -c "$charge_method" \
    -s "$verbose" \
    -nc "$net_charge" \
    -rn "$resname" \
    -at gaff2 \
    -pf "$intermediate_files"

echo "Created $output"
echo "Next: run parmchk2 -i $output -f mol2 -o ${stem}.frcmod"
