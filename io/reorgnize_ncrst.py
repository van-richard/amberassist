import os
import sys
import numpy as np

run = int(sys.argv[1])

if run == 0:
    index = np.loadtxt("results_00/plot.REX", dtype=int)[-1][1:] - 1
    for i, j in enumerate(index):
        os.rename("%02d/step6.00_equilibration.ncrst" % i, "%02d/step6.00_equilibration_orig.ncrst" % i)

    for i, j in enumerate(index):
        os.symlink("../%02d/step6.00_equilibration_orig.ncrst" % i, "%02d/step6.00_equilibration.ncrst" % j)
elif run == 1:
    index = np.loadtxt("results_01/plot.REX", dtype=int)[-1][1:] - 1
    for i, j in enumerate(index):
        os.rename("%02d/step6.01_equilibration.ncrst" % i, "%02d/step6.01_equilibration_orig.ncrst" % i)

    for i, j in enumerate(index):
        os.symlink("../%02d/step6.01_equilibration_orig.ncrst" % i, "%02d/step6.01_equilibration.ncrst" % j)
elif run == 2:
    index = np.loadtxt("results_02/plot.REX", dtype=int)[-1][1:] - 1
    for i, j in enumerate(index):
        os.rename("%02d/step6.02_equilibration.ncrst" % i, "%02d/step6.02_equilibration_orig.ncrst" % i)

    for i, j in enumerate(index):
        os.symlink("../%02d/step6.02_equilibration_orig.ncrst" % i, "%02d/step6.02_equilibration.ncrst" % j)
elif run > 2:
    index = np.loadtxt("results_%02d/plot_final.REX" % run, dtype=int)[-1][1:] - 1
    for i, j in enumerate(index):
        os.rename("%02d/step7.%02d_equilibration.ncrst" % (i, run - 3), "%02d/step7.%02d_equilibration_orig.ncrst" % (i, run - 3))

    for i, j in enumerate(index):
        os.symlink("../%02d/step7.%02d_equilibration_orig.ncrst" % (i, run - 3), "%02d/step7.%02d_equilibration.ncrst" % (j, run - 3))
