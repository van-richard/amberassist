#!/bin/bash
# Prepare a single QMMM+PLUMED simulation directory.
# Edit this file for the system, then run write_mdin.sh and gen_plumeddat.sh.

set -euo pipefail

REF="../dft"
init="step3_pbcsetup"
MDRST="prod00.ncrst"

THERMOSTAT="langevin" # langevin, sinr
NSTEPS5=1000
PRINTFREQ=1

####################
# PLUMED Configuration
####################
PLUMED_METHOD="metad" # metad, wtmetad
PLUMED_CV="2d"        # 2d, d1-d2

##################
# QM Configuration
##################
PROTEIN_RESIDUES="301,322"
METAL_RESIDUES="371"
WATER_RESIDUES="397,398,399,400"
NA_MASK="(:13&@385-409)|(:14&!@417-442)"

# Set QMMASK_OVERRIDE for a fully custom Amber mask; otherwise the named
# components above are combined into a default QM region.
QMMASK_OVERRIDE=""
QMMASK=${QMMASK_OVERRIDE:-(:${PROTEIN_RESIDUES}&!@N,H,CA,HA,C,O)|(${NA_MASK})|:${METAL_RESIDUES},${WATER_RESIDUES}}

QMTHEORY="EXTERN" # EXTERN for QMHub, or an Amber-supported semiempirical method
QMHUB_MODE="DFT"  # DFT or MTS; used only when QMTHEORY=EXTERN
QMCHARGE="+1"

####################
# Don't change this
####################
run_dir=$(realpath ..)
inp_dir="${run_dir}/input"

for v in THERMOSTAT NSTEPS5 PRINTFREQ PLUMED_METHOD PLUMED_CV QMMASK QMTHEORY QMCHARGE; do
    [ -n "${!v}" ] || { echo "ERROR: missing ${v}" >&2; exit 1; }
done

case "$THERMOSTAT" in
    langevin|sinr) ;;
    *) echo "ERROR: THERMOSTAT must be 'langevin' or 'sinr' (got '$THERMOSTAT')" >&2; exit 1 ;;
esac

case "$PLUMED_METHOD" in
    metad|wtmetad) ;;
    *) echo "ERROR: PLUMED_METHOD must be 'metad' or 'wtmetad' (got '$PLUMED_METHOD')" >&2; exit 1 ;;
esac

case "$PLUMED_CV" in
    2d|d1-d2) ;;
    *) echo "ERROR: PLUMED_CV must be '2d' or 'd1-d2' (got '$PLUMED_CV')" >&2; exit 1 ;;
esac

if [ "$QMTHEORY" = "EXTERN" ]; then
    case "$QMHUB_MODE" in
        DFT|MTS) ;;
        *) echo "ERROR: QMHUB_MODE must be 'DFT' or 'MTS' when QMTHEORY=EXTERN" >&2; exit 1 ;;
    esac
fi

cat <<_EOF > qm_info.txt
run_dir=${run_dir}
input_dir=${inp_dir}
protein=${PROTEIN_RESIDUES}
na=${NA_MASK}
metal=${METAL_RESIDUES}
water=${WATER_RESIDUES}
qmmask=${QMMASK}
qmtheory=${QMTHEORY}
qmhub_mode=${QMHUB_MODE}
qmcharge=${QMCHARGE}
thermostat=${THERMOSTAT}
nsteps5=${NSTEPS5}
print_freq=${PRINTFREQ}
plumed_method=${PLUMED_METHOD}
plumed_cv=${PLUMED_CV}
_EOF

#####################
# Get files from $REF
#####################
echo "copying: files from ${REF}"
cp "${REF}/${init}.parm7" "${run_dir}/"
cp "${REF}/${MDRST}" "${run_dir}/step5.00_equilibration_inp.ncrst"

echo "qmmask=${QMMASK}"
echo "qmcharge=${QMCHARGE}"
echo "plumed_method=${PLUMED_METHOD}"
echo "plumed_cv=${PLUMED_CV}"
if [ "$QMTHEORY" = "EXTERN" ]; then
    echo "qmhub_mode=${QMHUB_MODE}"
fi
