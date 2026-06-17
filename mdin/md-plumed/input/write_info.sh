#!/bin/bash
# Prepare a single classical MD+PLUMED simulation directory.
# Edit this file for the system, then run write_mdin.sh and gen_plumeddat.sh.

set -euo pipefail

REF="../dft"
init="step3_pbcsetup"
MDRST="prod00.ncrst"

THERMOSTAT="langevin" # langevin, sinr
NSTEPS=1000
PRINTFREQ=1

####################
# PLUMED Configuration
####################
PLUMED_METHOD="metad" # metad, wtmetad
PLUMED_CV="2d"        # 2d, d1-d2

####################
# Don't change this
####################
run_dir=$(realpath ..)
inp_dir="${run_dir}/input"

for v in THERMOSTAT NSTEPS PRINTFREQ PLUMED_METHOD PLUMED_CV; do
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

cat <<_EOF > md_info.txt
run_dir=${run_dir}
input_dir=${inp_dir}
thermostat=${THERMOSTAT}
nsteps=${NSTEPS}
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

echo "plumed_method=${PLUMED_METHOD}"
echo "plumed_cv=${PLUMED_CV}"
