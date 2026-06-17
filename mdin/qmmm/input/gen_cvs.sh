#!/bin/bash
set -euo pipefail

QMINFO="qm_info.txt"
LIST="../list"
[[ -f "$QMINFO" ]] || { echo "ERROR: missing $QMINFO" >&2; exit 1; }
[[ -f "$LIST" ]] || { echo "ERROR: missing $LIST; run write_mdin.sh first" >&2; exit 1; }

cwd=""
n_windows=""
cv_min=""
print_freq=""
rc=""

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue
  [[ "$line" != *"="* ]] && continue

  key="${line%%=*}"
  val="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"

  case "$key" in
    cwd) cwd="$val" ;;
    n_windows) n_windows="$val" ;;
    cv_min) cv_min="$val" ;;
    print_freq) print_freq="$val" ;;
    rc) rc="$val" ;;
  esac
done < "$QMINFO"

[[ -n "$cwd" ]] || { echo "ERROR: $QMINFO missing 'cwd='" >&2; exit 1; }
[[ -n "$n_windows" ]] || { echo "ERROR: $QMINFO missing 'n_windows='" >&2; exit 1; }
[[ -n "$cv_min" ]] || { echo "ERROR: $QMINFO missing 'cv_min='" >&2; exit 1; }
[[ -n "$rc" ]] || { echo "ERROR: $QMINFO missing 'rc='" >&2; exit 1; }

mapfile -t windows < "$LIST"
[[ "${#windows[@]}" -gt 0 ]] || { echo "ERROR: $LIST is empty" >&2; exit 1; }
if [[ "${#windows[@]}" -ne "$n_windows" ]]; then
  echo "ERROR: $LIST has ${#windows[@]} windows, but $QMINFO says n_windows=${n_windows}" >&2
  exit 1
fi

CV_i="$cv_min"
step="0.1"
inp_dir="../input"

for window in "${windows[@]}"; do
  echo "create: ${window}, ${inp_dir}/cv.rst"
  mkdir -p "../$window"
  cd "../$window"
  cp "${inp_dir}/cv.rst.tmp" "cv.rst"

  nn="$(printf "%.3f" "${CV_i}")"
  sed -i "s/__RC__/${rc}/g" cv.rst
  sed -i "s/__RST__/${nn}/g" cv.rst

  CV_i="$(echo "${CV_i} + ${step}" | bc)"
  cd - >/dev/null
done
