#!/bin/bash

module load ambertools23

#sys="4nt"
sys=$1
base="../1leader/${sys}"
init="step3_pbcsetup_1264"
atommask="@CA"
n_vecs="10"

results="pca2/${sys}"
mkdir -p ${results}

cpptraj <<_EOF
parm ${base}ds/${init}.parm7 [ds]
parm ${base}ov/${init}.parm7 [ov]

trajin ${base}ds/prod??.nc parm [ds]
trajin ${base}ov/prod??.nc parm [ov]

rms first ${atommask}
strip :WAT,Na+,Cl-,MG,DA5,DA,DA3,DT5,DT,DT3,DG5,DG,DG3,DC5,DC,DC3,
average crdset step3-average
createcrd step3-trajectories
run

crdaction step3-trajectories rms ref step3-average ${atommask}
crdaction step3-trajectories matrix covar name step3-covar ${atommask}

runanalysis diagmatrix step3-covar out ${results}/step3-evecs.dat \
vecs ${n_vecs} name myEvecs \
nmwiz nmwizvecs ${n_vecs} nmwizfile ${results}/step3.nmd nmwizmask ${atommask}
run
clear all

readdata ${results}/step3-evecs.dat name Evecs

parm ${base}ds/${init}.parm7
parmstrip !(${atommask})
parmwrite out ${results}/step3-mode.parm7

runanalysis modes name Evecs trajout ${results}/step3-mode1.nc \
pcmin -100 pcmax 100 tmode 1 trajoutmask ${atommask} trajoutfmt netcdf
_EOF
