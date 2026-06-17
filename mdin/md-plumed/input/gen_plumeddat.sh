#!/bin/bash
# Select a PLUMED template and write ../plumed.dat.

set -euo pipefail

MDINFO="md_info.txt"
OUT="../plumed.dat"
[[ -f "$MDINFO" ]] || { echo "ERROR: missing $MDINFO" >&2; exit 1; }

plumed_method=""
plumed_cv=""

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  val="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"

  case "$key" in
    plumed_method) plumed_method="$val" ;;
    plumed_cv) plumed_cv="$val" ;;
  esac
done < "$MDINFO"

plumed_method="${PLUMED_METHOD:-${plumed_method:-metad}}"
plumed_cv="${PLUMED_CV:-${plumed_cv:-2d}}"
template="plumed.${plumed_method}.${plumed_cv}.dat"

[[ -f "$template" ]] || {
  echo "ERROR: missing PLUMED template: $template" >&2
  echo "Available templates:" >&2
  find . -maxdepth 1 -name 'plumed.*.dat' -printf '  %f\n' | sort >&2
  exit 1
}

cp "$template" "$OUT"
echo "created: $OUT from $template"
