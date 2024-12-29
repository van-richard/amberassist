#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -n 40
#SBATCH --cpus-per-task=16
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -t 5-00:00:00
#SBATCH -J asm

date

#source oscer.env
source pete.env
pRUN="00"
RUN="01"

echo ${nwindows} > guess${RUN}
cat results_${pRUN}/0_final.string >> guess${RUN}
awk {'print $1'} results_${pRUN}/final_parameters.dat > parameters${RUN}.dat
awk {'print $2'} results_${pRUN}/final_parameters.dat >> parameters${RUN}.dat

cat <<_EOF > STRING.BAK
\$STRINGSETUP
dir = "results/"
guess_file = "guess${RUN}"
!params_file = "parameters${RUN}.dat"
preparation_steps = 0
REX_period = 100
fix_ends = .true.
!read_M = .true.
only_PMF = .false.
!force_constant = 250.0
!force_constant_d = 50.0
\$END
_EOF

mkdir results
cp STOP_STRING results
cp STRING.BAK STRING
bash gen_groupfile.sh ${RUN} > groupfile
$SANDER -ng ${nwindows} -groupfile groupfile
mv $OUTDIR ${OUTDIR}_$RUN
python reorgnize_ncrst.py $RUN
ln -sf ${OUTDIR}_${RUN}/final_parameters.dat .
ln -sf ${OUTDIR}_${RUN}/0_final.string .

# echo 80 > guess2
# cat results_01/0_final.string >> guess2
# awk {'print $1'} results_01/final_parameters.dat > parameters2.dat
# awk {'print $2'} results_01/final_parameters.dat >> parameters2.dat
# 
# mkdir results
# cp STOP_STRING results
# mv STRING.BAK STRING
# bash gen_groupfile.sh 02 > groupfile
# $SANDER -ng 80 -groupfile groupfile
# mv results results_02
# python reorgnize_ncrst.py 2
# ln -s results_02/final_parameters.dat .
# ln -s results_02/0_final.string .
# sleep 48h

#echo 40 > guess4
#cat results_02/0_final.string >> guess4
#awk {'print $1'} results_02/final_parameters.dat > parameters4.dat
#awk {'print $2'} results_02/final_parameters.dat >> parameters4.dat
#
#mkdir results
#cp STOP_STRING results
#mv STRING.BAK STRING
#bash gen_groupfile.sh 03 > groupfile
#$SANDER -ng 40 -groupfile groupfile
#mv results results_03
#python reorgnize_ncrst.py 3
#ln -s results_03/final_parameters.dat .
#ln -s results_03/0_final.string .

#echo 40 > guess5
#cat results_03/0_final.string >> guess5
#awk {'print $1'} results_03/final_parameters.dat > parameters5.dat
#awk {'print $2'} results_03/final_parameters.dat >> parameters5.dat
#
#mkdir results
#cp STOP_STRING results
#mv STRING.BAK STRING
#bash gen_groupfile.sh 04 > groupfile
#$SANDER -ng 40 -groupfile groupfile
#mv results results_04
#python reorgnize_ncrst.py 4
#ln -s results_04/final_parameters.dat .
#ln -s results_04/0_final.string .

# mkdir results
# for run in `seq -f"%02g" 0 0`; do
#     bash gen_groupfile2.sh $run > groupfile
#     $SANDER -ng 60 -groupfile groupfile
#     mv results results_01
#     python reorgnize_ncrst.py 1
# done
