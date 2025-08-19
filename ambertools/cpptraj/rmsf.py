import os
import sys
import argparse
import pytraj as pt
import matplotlib.pyplot as plt
import numpy as np

pname="step3_pbcsetup" # Topology file
tname='prod00' # Trajectory file 
figname = "rmsf"

ambermask='@CA' # Atom mask selection

parm = f'{pname}.parm7'
cord = f'{tname}.nc'

os.makedirs('img', exist_ok=True)
os.makedirs('raw_data', exist_ok=True)

# Load trajectory
traj = pt.iterload(cord, top=parm)

# Superimpose to 1st frame and alpha carbons
pt.superpose(traj, ref=0, mask=ambermask)

data = pt.rmsf(traj, mask=ambermask)

resnum = len(data.T[0]) + 1
xdata = np.arange(1, resnum)
ydata = data.T[1]

# Plot Simulation Time vs RMSD
plt.plot(xdata, ydata)
plt.xlabel('Residue Number')
plt.ylabel('RMSF (Å)')

plt.savefig('img/{figname}', dpi=300)

