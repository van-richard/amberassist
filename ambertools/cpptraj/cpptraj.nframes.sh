#!/bin/bash

module load ambertools23

init="step3_pbcsetup"

cpptraj -p ${init}.parm7 -y ${init}.rst7 -tl
