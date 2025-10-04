import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from qmhub.units import AMBER_HARTREE_TO_KCAL, AMBER_BOHR_TO_A

n_windows = 42
qmtheory = "b3lyp_6-31+gd"

# combine energy of all frames in window
for window in range(n_windows):
    print(f"window: {window}")
    fnames = sorted(glob('../%02d/wtp/*/qmmm_%s_energy.npy' % (window, qmtheory)))
    energy = []
    for fname in fnames:
        f = np.load(fname)
        energy.append(f)
    energies = np.concatenate(energy)
    np.save('../%02d/wtp_%s_energy.npy' % (window, theory), energies)

# combine energy of all windows in project
fnames = sorted(glob(f"../??/wtp_{qmtheory}_energy.py"))
ene = [np.load(f)[::] for f in fnames[:]]
energies = np.column_stack(ene)
np.save(f"qmmm_{qmtheory}_energy.npy", np.swapaxes(energies, 0, 1))

