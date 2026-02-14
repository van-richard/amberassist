#!/bin/bash
# Prepare QMMM free energy simulations 
# Umbrella sampling 

inp_dir="../input"
total_w=$(cat ../list | wc -l)
n_windows=$((${total_w}-1))

for i in $(seq 0 ${n_windows}); do
    printf -v window "%02d" $i
    mkdir -p ../$window
    cd ../$window

    ln -sf ${inp_dir}/${init}.parm7 .
    cp ${inp_dir}/step5.00_equilibration.mdin .
    cp ${inp_dir}/step6.00_equilibration.mdin .
    
    # Setup MD input for forward pull
    if [ ${window} == "00" ]; then
        ln -sf ${inp_dir}/${MDRST}  step5.00_equilibration_inp.ncrst
        IREST=0
        NTX=1
    else
        printf -v pstep "%02d" $((${i}-1))
        ln -sf ../${pstep}/step5.00_equilibration.ncrst step5.00_equilibration_inp.ncrst
        IREST=1
        NTX=5
    fi

    sed -i "s/__IREST__/${IREST}/;s/__NTX__/${NTX}/" step5.00_equilibration.mdin
    
    # Setup MD input for reverse pull
    printf -v last_window "%02d" $((${i}+1))
    if [ ${window} == "00" ]; then
        sed "s/0/${IREST}/;s/1/${NTX}/;s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    else
        sed "s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    fi

    if [ ${window} != $(tail -n 1 ../list) ]; then
        ln -sf ../${last_window}/step5.01_equilibration.ncrst step5.01_equilibration_inp.ncrst
    else
        ln -sf step5.00_equilibration.ncrst step5.01_equilibration_inp.ncrst
    fi
     
    
    for STEP in "step5.00" "step6.00" "step5.01"; do
        sed -i "s/__QMHUBSCRATCH__/${QMHUBSCRATCH}/" ${STEP}_equilibration.mdin
    done

    cd -
done

