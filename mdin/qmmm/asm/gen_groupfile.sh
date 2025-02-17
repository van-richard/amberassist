#!/bin/bash


init="step3_pbcsetup"

i=$(expr $1 + 0) #Removing leading zeros

if [ $i -eq 0 ]; then
    pstep="step5.00_equilibration"
else
    (( j = i - 1 ))
    pstep=$(printf "step6.%02d_equilibration" $j)
fi

istep=$(printf "step6.%02d_equilibration" $i)

for w in $(cat list); do
    echo "-O -i ${w}/${istep}.mdin -o ${w}/${istep}.mdout -p input/${init}.parm7 -c ${w}/${pstep}.ncrst -r ${w}/${istep}.ncrst -x ${w}/${istep}.nc -ref input/${init}.ncrst -inf ${w}/${istep}.mdinfo"
done
