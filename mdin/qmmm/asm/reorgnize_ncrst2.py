import os
import sys
import numpy as np

rep = int(sys.argv[1])
runs = 10

for run in range(runs):
    if rep == run:
        index = np.loadtxt(f"results_{rep:02d}/plot.REX", dtype=int)[-1][1:] - 1
        for i, j in enumerate(index):
            os.rename(f"{i:02d}/step6.{rep:02d}_equilibration.ncrst" , f"{i:02d}/step6.{rep:02d}_equilibration_orig.ncrst")

        for i, j in enumerate(index):
            os.symlink(f"../{i:02d}/step6.{rep:02d}_equilibration_orig.ncrst", f"{i:02d}/step6.{rep:02d}_equilibration.ncrst")
#     elif step > runs:
#         index = np.loadtxt(f"results_{step:02d}/plot_final.REX", dtype=int)[-1][1:] - 1
#         for i, j in enumerate(index):
#             os.rename(f"{i:02d}/step7.%02d_equilibration.ncrst" % (i, run - 3), "%02d/step7.%02d_equilibration_orig.ncrst" % (i, run - 3))
# 
#         for i, j in enumerate(index):
#             os.symlink(f"../{i:02d}/step7.{step-len(runs):02d}_equilibration_orig.ncrst", f"{step:02d}/step7.{step-len(runs):02d}_equilibration.ncrst" % (j, run - 3))
