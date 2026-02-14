#!/bin/bash
# Prepare QMMM free energy simulations 
# Umbrella sampling 

REF="../dft"
init="step3_pbcsetup"
MDRST="prod00.ncrst"

N_WINDOWS=42
CV_MIN=-1.90
RC="409,410,410,6255"

THERMOSTAT="langevin" # langevin, sinr
NSTEPS5=500
NSTEPS6=800
PRINTFREQ=50

##################
# QM Configuration
##################
PROTEIN="301,322"
METAL="371"
WATER="397,398,399,400"
NA="(:13&@385-409)|(:14&!@417-442)"
QMMASK="(:${PROTEIN}&!@N,H,CA,HA,C,O)|(${NA})|:${METAL},${WATER}"
QMTHEORY="EXTERN" #QMHub or give semi-empirical method 
QMCHARGE="+1"

####################
# Don't change this
####################
cwd=$(realpath ..)
inp_dir="${cwd}/input"


cat <<_EOF > qm_info.txt
cwd=${cwd}
protein=${PROTEIN}
na=${NA}
metal=${METAL}
water=${WATER}
qmmask=${QMMASK}
qmtheory=${QMTHEORY}
qmcharge=${QMCHARGE}
thermostat=${THERMOSTAT}
nsteps5=${NSTEPS5}
nsteps6=${NSTEPS6}
n_windows=${N_WINDOWS}
cv_min=${CV_MIN}
print_freq=${PRINTFREQ}
rc=${RC}
_EOF


#####################
# Get files from $REF
#####################
echo "copying: files from ${REF}"
cp ${REF}/${init}.parm7 .
cp ${REF}/${MDRST} .

