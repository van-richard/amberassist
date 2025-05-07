#!/usr/bin/env python
# coding: utf-8

# # Pairwise RMSD (2D-RMSD)
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('~/Scripts/bin/v.mplstyle')

import pytraj as pt

# Load trajectory and topology files
# relativbe path tooooo
traj = pt.iterload('../1leader/4ntds/prod00.nc', top='../1leader/4ntds/step3_pbcsetup_1264.parm7')
traj # Same as print(traj)


data = pt.pairwise_rmsd(traj, mask='@CA')

im = plt.imshow(data)

plt.colorbar(im, label='2D-RMSD (Å)')
plt.gca().invert_yaxis() # invert y-axis
plt.xlabel('Frame Number')
plt.ylabel('Frame Number')

os.makedirs('img', exist_ok=True)
# plt.savefig('img/2drmsd-4ntds.png') # uncomment to save


