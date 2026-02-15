import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
from sklearn.utils import resample

import pymbar
from pymbar.mbar_pmf import mbar_pmf

os.makedirs('img', exist_ok=True)

steprep=sys.argv[1]
step = steprep.split('.')[0]
rep = steprep.split('.')[1]

n_windows = int(sys.argv[2])
val_min = float(sys.argv[3]) # restraint at window 00
val_max = float(sys.argv[4]) # restraint at window 41

val_step = 0.1
fc = 300.0      # forces constant for restraint - AMBERVALUE*2 - Richard had AMBERVALUE=150.0
nbins = n_windows - 1

opa=0.4
_xfrac=0.05
_yfrac=0.9

val0_k = np.arange(val_min, val_max+val_step, val_step)
K_k = np.ones(n_windows) * fc



#initial = np.loadtxt(f"freefiles/freefile_mbar_{step}.{rep}")

fnames = sorted(glob('freefiles/freefile_mbar_*'))

fig,axs = plt.subplots(figsize=(10,6), dpi=120)

for i in range(len(fnames)):

    t_range = str(fnames[i].split('_')[-1])
    initial = np.loadtxt(fnames[i])

    xdata=initial[:,0]
    ydata=initial[:,1] - initial[:10,1].min()
    edata=initial[:,2]

    dgd = round(initial[:,1].max() - initial[:10,1].min(),1) # Delta G daggerV
    err = round(initial[initial[:,1].argmax()][2], 1) # mbar error

    initial_label = f"{t_range}: $\Delta G^\ddag$ = {dgd} $\pm$ {err}"

    axs.errorbar(xdata, ydata, yerr=edata, linewidth=1, alpha=opa+0.1, label=initial_label)
    
    axs.grid(linestyle='-', alpha=opa-0.2)
    
    #axs.annotate(f"$\Delta G^\ddag$ = {dgd} $\pm$ {err}", 
    #           xy=(_xfrac,_yfrac-0.15), xycoords='axes fraction',bbox=dict(fc='w', alpha=opa+0.2))

    axs.set_xlabel("r1 - r2 (Å)")
    axs.set_ylabel("Potential of Mean Force (kcal/mol)")

plt.legend()

plt.savefig(f"img/prelim-B-{step}.{rep}.png")

