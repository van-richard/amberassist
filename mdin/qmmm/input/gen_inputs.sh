#!/bin/bash
# Prepare QMMM free energy simulation window inputs.

set -euo pipefail

inp_dir="../input"
list_file="../list"
init="step3_pbcsetup"
MDRST="prod00.ncrst"
QMINFO="${inp_dir}/qm_info.txt"

[ -f "$list_file" ] || { echo "ERROR: missing $list_file; run write_mdin.sh first" >&2; exit 1; }
[ -f "$QMINFO" ] || { echo "ERROR: missing $QMINFO" >&2; exit 1; }

for required_file in \
    "${inp_dir}/${init}.parm7" \
    "${inp_dir}/${MDRST}" \
    "${inp_dir}/step5.00_equilibration.mdin" \
    "${inp_dir}/step6.00_equilibration.mdin"; do
    [ -f "$required_file" ] || { echo "ERROR: missing required input: $required_file" >&2; exit 1; }
done

qmtheory=""
qmhub_mode=""
cwd=""

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    [ "$line" = "${line#*=}" ] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    case "$key" in
        cwd) cwd="$val" ;;
        qmtheory) qmtheory="$val" ;;
        qmhub_mode) qmhub_mode="$val" ;;
    esac
done < "$QMINFO"

[ -n "$qmtheory" ] || { echo "ERROR: $QMINFO missing 'qmtheory='" >&2; exit 1; }
cwd="${cwd:-$(realpath ..)}"

if [ "$qmtheory" = "EXTERN" ]; then
    qmhub_mode="${qmhub_mode:-DFT}"
    case "$qmhub_mode" in
        DFT) qmhub_ini="qmhub_dft.ini" ;;
        MTS) qmhub_ini="qmhub_mts.ini" ;;
        *) echo "ERROR: qmhub_mode must be 'DFT' or 'MTS' when qmtheory=EXTERN (got '$qmhub_mode')" >&2; exit 1 ;;
    esac
    [ -f "${inp_dir}/${qmhub_ini}" ] || { echo "ERROR: missing ${inp_dir}/${qmhub_ini}" >&2; exit 1; }
fi

mapfile -t windows < "$list_file"
[ "${#windows[@]}" -gt 0 ] || { echo "ERROR: $list_file is empty" >&2; exit 1; }

esc() { printf '%s' "$1" | sed 's/[\\&#]/\\&/g'; }

last_window="${windows[$((${#windows[@]} - 1))]}"
for idx in "${!windows[@]}"; do
    window="${windows[$idx]}"
    mkdir -p "../$window"
    cd "../$window"

    ln -sf "${inp_dir}/${init}.parm7" .
    cp "${inp_dir}/step5.00_equilibration.mdin" .
    cp "${inp_dir}/step6.00_equilibration.mdin" .

    # Setup MD input for forward pull
    if [ "$idx" -eq 0 ]; then
        ln -sf "${inp_dir}/${MDRST}" step5.00_equilibration_inp.ncrst
        IREST=0
        NTX=1
    else
        pstep="${windows[$((idx - 1))]}"
        ln -sf "../${pstep}/step5.00_equilibration.ncrst" step5.00_equilibration_inp.ncrst
        IREST=1
        NTX=5
    fi

    sed -i "s/__IREST__/${IREST}/;s/__NTX__/${NTX}/" step5.00_equilibration.mdin

    # Setup MD input for reverse pull
    if [ "$idx" -eq 0 ]; then
        sed "s/0/${IREST}/;s/1/${NTX}/;s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    else
        sed "s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    fi

    if [ "$window" != "$last_window" ]; then
        next_window="${windows[$((idx + 1))]}"
        ln -sf "../${next_window}/step5.01_equilibration.ncrst" step5.01_equilibration_inp.ncrst
    else
        ln -sf step5.00_equilibration.ncrst step5.01_equilibration_inp.ncrst
    fi

    if [ "$qmtheory" = "EXTERN" ]; then
        ln -sf "${inp_dir}/${qmhub_ini}" qmhub.ini
        qmhubscratch="${QMHUBSCRATCH:-/tmp/${USER}/$(basename "$cwd")/${window}/qmhub}"
        qmhubscratch_esc=$(esc "$qmhubscratch")
        for STEP in "step5.00" "step6.00" "step5.01"; do
            sed -i "s#__QMHUBSCRATCH__#${qmhubscratch_esc}#g" "${STEP}_equilibration.mdin"
        done
    fi

    cd - >/dev/null
done
