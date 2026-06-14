import sys
from glob import glob

import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.integrate import cumtrapz
from parmed.amber import NetCDFTraj, AmberParm
from qmhub.units import AMBER_HARTREE_TO_KCAL, AMBER_BOHR_TO_A


def mean_force_otter(pos, force):
    f1 = np.dot(force[0] - force[1], pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])
    f2 = np.dot(force[3] - force[2], pos[3] - pos[2]) / np.linalg.norm(pos[3] - pos[2])
    return (f1 - f2) / 4


def get_pmf(forces, positions):
    mean_forces = []
    for force, pos in zip(forces, positions):
        mean_force = np.zeros(len(pos))
        for j in range(len(pos)):
            mean_force[j] = mean_force_otter(pos[j], force[j])
        mean_forces.append(mean_force)
    mean_forces = np.array(mean_forces)
    mean = np.array(mean_forces).mean(axis=1)
    mean_var = mean_forces.var(axis=1) / mean_forces.shape[1]
    return cumtrapz(mean, dx=0.05), np.sqrt(cumtrapz(mean_var, dx=0.05))


n_windows = 80
num_prot_atoms = 5745
qm_index2 = np.array([0, 9, 5, 7])
qm_index = qm_index2 + num_prot_atoms
num_prot_atoms = qm_index[0]
parm = AmberParm("../../input/step3_pbcsetup.parm7")

pos_all = []
for i in range(n_windows):
    pos = []
    for j in range(4, 10):
        traj = NetCDFTraj.open_old("../../%02d/step6.%02d_equilibration.nc" % (i, j))
        pos.append(traj.coordinates[:, qm_index])
        traj.close()

    pos_all.append(np.concatenate(pos))

mm_atom_forces = []
for i in range(n_windows):
    force1 = -1 * np.load("../%02d/gas_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    force2 = -1 * np.load("../%02d/qmmm_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    mm_atom_forces.append(0.5 * (force1 + force2))
mm_atom_forces = np.array(mm_atom_forces)

mm_barrier_pot = np.zeros((num_prot_atoms, 2))

for i in range(num_prot_atoms):
    pmf_mm_atom, pmf_mm_atom_std = get_pmf(mm_atom_forces[:, :, :, :, i], pos_all)
    barrier_mm_atom, barrier_mm_atom_std = pmf_mm_atom[28] - pmf_mm_atom[1], pmf_mm_atom_std[28]
    mm_barrier_pot[i] = barrier_mm_atom / parm.atoms[i].charge, barrier_mm_atom_std / parm.atoms[i].charge

np.save("mm_barrier_pot", mm_barrier_pot)
np.savetxt("mm_barrier_pot.dat", mm_barrier_pot, fmt="%5.2f")
