#!/bin/bash
#

steprep="step6.0?"
n_windows=$(cat ../list | wc -l)
cv_min="-1.90"
cv_max="2.20"

t_total="30.0"      # first 20 ps
t_stepsize="5.0"    # compute PMF from this datasize
t_shift="5.0"       # shift sliding window size
t_start="0.0"       # time start


python write_cvs.py ${steprep} ${n_windows}

python mbar_sliding.py ${steprep} ${n_windows} \
    ${cv_min} ${cv_max} \
    ${t_total} ${t_stepsize} ${t_shift} ${t_start} \
    2>&1 | tee mbar_sliding.log

python plot_mbar_sliding.py ${steprep} ${n_windows} \
    ${cv_min} ${cv_max} 

