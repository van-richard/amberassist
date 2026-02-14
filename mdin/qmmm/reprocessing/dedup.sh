#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

dryrun=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dryrun=1
  shift
fi

# Run from window dir; default target folder is qmhub2/
dir="${1:-qmhub2}"
prefix="qmmm.inp_"

cd "$dir" || { echo "ERROR: cannot cd into '$dir'"; exit 1; }

run() {
  if (( dryrun )); then
    echo "+ $*"
  else
    "$@"
  fi
}

files=( "${prefix}"* )
((${#files[@]})) || { echo "ERROR: no files matching ${prefix}* in $dir"; exit 1; }

max=-1
width=0

for f in "${files[@]}"; do
  base="${f##*/}"
  n="${base#${prefix}}"
  [[ "$n" =~ ^[0-9]+$ ]] || continue
  (( ${#n} > width )) && width=${#n}
  val=$((10#$n))                # force base-10 (avoid octal)
  (( val > max )) && max=$val
done

(( max >= 0 )) || { echo "ERROR: couldn't parse any numeric indices in ${prefix}*"; exit 1; }

# If max is odd, last pair is incomplete; drop it with a warning
if (( max % 2 == 1 )); then
  echo "WARNING: max index is odd ($max); ignoring last unpaired file"
  max=$((max - 1))
fi

# Helper for consistent zero-padding based on observed width
fname() { printf "%s%0*d" "$prefix" "$width" "$1"; }

unique_count=$(( max / 2 + 1 ))
echo "Found max index: $max  -> will keep $unique_count unique frames"

# 1) Remove odd duplicates first (so renames won't collide)
for ((idx=1; idx<=max; idx+=2)); do
  f="$(fname "$idx")"
  [[ -e "$f" ]] && run rm -f -- "$f"
done

# 2) Renumber even indices down by factor of 2 (skip 0 -> 0)
for ((idx=2; idx<=max; idx+=2)); do
  src="$(fname "$idx")"
  dst="$(fname "$((idx/2))")"

  [[ -e "$src" ]] || { echo "WARNING: missing $src (skipping)"; continue; }

  if [[ -e "$dst" ]]; then
    echo "ERROR: destination exists: $dst (refusing to overwrite)"
    exit 1
  fi

  run mv -- "$src" "$dst"
done

echo "Done. Kept $unique_count frames in $(pwd)"

