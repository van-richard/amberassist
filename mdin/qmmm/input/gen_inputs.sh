#!/bin/bash
# Prepare QMMM free energy simulations 
# Umbrella sampling 


echo "create: ${cwd}/list"
cd ${cwd}
seq -w 0 ${n_windows} > list

seq 0 ${n_windows} | while read i; do
    printf -v window "%02d" $i
    mkdir -p $window
    cd $window

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
    if [ ${window} == "00" ]; then
        sed "s/0/${IREST}/;s/1/${NTX}/;s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    else
        sed "s/step5.00/step5.01/" step5.00_equilibration.mdin > step5.01_equilibration.mdin
    fi
    
    for STEP in "step5.00" "step6.00" "step5.01"; do
        sed -i "\
            s/__QMMASK__/${QMMASK}/;\
            s/__QMCHARGE__/${QMCHARGE}/;\
            s/__QMTHEORY__/${QMTHEORY}/;\
            s/__QMSHAKE__/${QMSHAKE}/;\
            s/__QMCUT__/${QMCUT}/;\
            s/__QMEWALD__/${QMEWALD}/;\
            s/__QMPME__/${QMPME}/;\
            s/__QMSWITCH__/${QMSWITCH}/;\
            s/__QMHUBSCRATCH__/${QMHUBSCRATCH}/" ${STEP}_equilibration.mdin
    done

    cd $cwd
    echo $(pwd)
done

date
