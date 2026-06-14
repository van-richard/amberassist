#!/usr/bin/env bash
set -euo pipefail

tmpl_dir="${1:-../reprocess}"              # passed as "../reprocess" or an absolute reprocess path
tmpl_name="${2:-step7_reprocess.mdin}"     # output mdin name in the current window directory
window="${3:-$(basename "$PWD")}"          # umbrella window name, e.g. "00"

tmpl="${tmpl_dir}/${tmpl_name}.tmp"
out="${tmpl_name}"
qm_info="../input/qm_info.txt"             # relative to a window dir: ../00 -> ../input
fallback_mdin="step6.00_equilibration.mdin"

[ -f "$tmpl" ] || { echo "ERROR: missing template: $tmpl"; exit 1; }
[[ "$window" =~ ^[0-9][0-9]$ ]] || { echo "ERROR: unexpected window name: $window"; exit 1; }
if [[ -e "$out" && "${REPROCESS_OVERWRITE_MDIN:-0}" != "1" ]]; then
  echo "ERROR: output already exists: $out"
  echo "Set REPROCESS_OVERWRITE_MDIN=1 to regenerate it intentionally."
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

strip_mdin_value() {
  local value
  value="$(trim "$1")"
  value="${value%,}"
  value="$(trim "$value")"
  if [[ "$value" == \'*\' ]]; then
    value="${value#\'}"
    value="${value%\'}"
  fi
  printf '%s' "$value"
}

read_qm_info_file() {
  local source="$1"
  qmmask=$(sed -n 's/^[[:space:]]*qmmask[[:space:]]*=[[:space:]]*//p' "$source" | head -n1)
  qmcharge=$(sed -n 's/^[[:space:]]*qmcharge[[:space:]]*=[[:space:]]*//p' "$source" | head -n1)
}

read_mdin_file() {
  local source="$1"
  qmmask=$(sed -n 's/^[[:space:]]*qmmask[[:space:]]*=[[:space:]]*//p' "$source" | head -n1)
  qmcharge=$(sed -n 's/^[[:space:]]*qmcharge[[:space:]]*=[[:space:]]*//p' "$source" | head -n1)
  qmmask="$(strip_mdin_value "$qmmask")"
  qmcharge="$(strip_mdin_value "$qmcharge")"
}

if [[ -f "$qm_info" ]]; then
  read_qm_info_file "$qm_info"
  qm_source="$qm_info"
else
  [ -f "$fallback_mdin" ] || { echo "ERROR: missing qm info: $qm_info and fallback mdin: $fallback_mdin"; exit 1; }
  # Metadata-only fallback: recover the current window's QM region and charge.
  read_mdin_file "$fallback_mdin"
  qm_source="$fallback_mdin"
fi

[ -n "$qmmask" ]   || { echo "ERROR: qmmask not found in $qm_source"; exit 1; }
[ -n "$qmcharge" ] || { echo "ERROR: qmcharge not found in $qm_source"; exit 1; }

# QMHub writes per-frame inputs into the real window path for later qmhub.squashfs.
if command -v realpath >/dev/null 2>&1; then
  window_dir="$(realpath .)"
else
  window_dir="$(pwd -P)"
fi
qmbase="${window_dir}/qmhub"

# Escape for sed replacement with '#' delimiter: escape \, &, #
esc() { printf '%s' "$1" | sed 's/[\\&#]/\\&/g'; }

qmbase_esc=$(esc "$qmbase")
qmmask_esc=$(esc "$qmmask")
qmcharge_esc=$(esc "$qmcharge")

sed -e "s#__QMBASE__#${qmbase_esc}#g" \
    -e "s#__QMMASK__#${qmmask_esc}#g" \
    -e "s#__QMCHARGE__#${qmcharge_esc}#g" \
    "$tmpl" > "$out"

# Catches empty template or bad read.
[ -s "$out" ] || { echo "ERROR: Output file $out is empty"; exit 1; }
