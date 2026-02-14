#!/bin/bash
#

if [ -f "step3_pbcsetup_1264.parm7" ]; then
    init="step3_pbcsetup_1264"
else
    init="step3_pbcsetup"
fi

mkdir -p cpptraj

cpptraj <<_EOF
parm ${init}.parm7
trajin prod??.nc
autoimage
trajout cpptraj/prod_all.nc
go

_EOF
