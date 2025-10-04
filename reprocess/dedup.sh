#!/usr/bin/env bash
# Remove duplicated files (qmhub/qmmm.inp_????)
# Pattern observed: for each i, we have j=2*i (keep -> rename to i) and k=2*i+1 (delete)

set -euo pipefail
shopt -s nullglob

i=0
while :; do
  j=$(printf "qmmm.inp_%04d" $(( i * 2     )))
  k=$(printf "qmmm.inp_%04d" $(( i * 2 + 1 )))

  # Stop when there is no next "even" file to compact
  [[ -e "$j" ]] || break

  target=$(printf "qmmm.inp_%04d" "$i")
  echo "renaming: $j -> $target ; deleting: $k (if present)"
  mv -f -- "$j" "$target"
  [[ -e "$k" ]] && rm -f -- "$k"

  ((i++))
done

echo "Compacted $i frames."

