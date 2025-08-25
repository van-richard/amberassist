#!/bin/bash

date

#module load amber/qmhub/torchmdnet
module load ambertools23

w=$1 # window

printf -v window "%02d" ${w}

cd $window


cpptraj -p step3_pbcsetup.parm7 -y step6.0*_equilibration.nc  -x all_step6.nc

date

