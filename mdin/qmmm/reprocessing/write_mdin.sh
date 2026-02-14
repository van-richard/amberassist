#!/bin/bash
set -euo pipefail

tmpl_dir="${1:-../reprocess}"                 # passed as "../reprocess"
tmpl_name="${2:-step7_reprocess.mdin}"     # passed as "step7_reprocess.mdin"
window="${3:-$(basename "$PWD")}"             # passed as "${window}"

tmpl="${tmpl_dir}/${tmpl_name}.tmp"
out="${tmpl_name}"                            # write step7_reprocess.mdin in current dir
qm_info="../input/qm_info.txt"                # relative to window dir (../00 -> ../input)

[ -f "$tmpl" ] || { echo "ERROR: missing template: $tmpl"; exit 1; }
[ -f "$qm_info" ] || { echo "ERROR: missing qm info: $qm_info"; exit 1; }

qmmask=$(sed -n 's/^[[:space:]]*qmmask[[:space:]]*=[[:space:]]*//p'   "$qm_info" | head -n1)
qmcharge=$(sed -n 's/^[[:space:]]*qmcharge[[:space:]]*=[[:space:]]*//p' "$qm_info" | head -n1)

[ -n "$qmmask" ]   || { echo "ERROR: qmmask not found in $qm_info"; exit 1; }
[ -n "$qmcharge" ] || { echo "ERROR: qmcharge not found in $qm_info"; exit 1; }

# QMBASE like your original: USER / project / window
project=$(basename "$(dirname "$PWD")")
qmbase="${USER}/${project}/${window}"

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

