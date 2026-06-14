#!/usr/bin/env python
import sys
import os
from glob import glob

import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.integrate import cumtrapz
from parmed.amber import AmberParm
from qmhub.units import AMBER_HARTREE_TO_KCAL, AMBER_BOHR_TO_A

def vector_cosine(vector1, vector2):
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    scalar_input = False
    if vector1.ndim == 1:
        vector1 = vector1[None]  # Makes vector1 2D
        vector2 = vector2[None]  # Makes vector2 2D
        scalar_input = True

    ret = np.zeros(len(vector1))
    for i, (v1, v2) in enumerate(zip(vector1, vector2)):
        ret[i] = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    if scalar_input:
        return np.squeeze(ret)
    return ret

def mean_force_otter(pos, force):
    f1 = np.dot(force[0] - force[1], pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])
    f2 = np.dot(force[2] - force[1], pos[2] - pos[1]) / np.linalg.norm(pos[2] - pos[1])
    cosa = vector_cosine(pos[0] - pos[1], pos[2] - pos[1])
    return (f1 - f2) / (4 - 2 * cosa)

# r1 should be the longer distance at the reactant side

def mean_force_otter_geom(pos):
    r1 = np.linalg.norm(pos[0] - pos[1])
    r2 = np.linalg.norm(pos[2] - pos[1])
    cosa = vector_cosine(pos[0] - pos[1], pos[2] - pos[1])
    return (4 + (1 - cosa**2) / (2 - cosa)) * (1 / r1 - 1 / r2) / (4 - 2 * cosa) / 1.677399001146518


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
    return cumtrapz(mean, dx=0.1), np.sqrt(cumtrapz(mean_var, dx=0.1))


def get_pmf_geom(positions):
    mean_forces = []
    for pos in positions:
        mean_force = np.zeros(len(pos))
        for j in range(len(pos)):
            mean_force[j] = mean_force_otter_geom(pos[j])
        mean_forces.append(mean_force)
    mean_forces = np.array(mean_forces)
    mean = np.array(mean_forces).mean(axis=1)
    mean_var = mean_forces.var(axis=1) / mean_forces.shape[1]
    return cumtrapz(mean, dx=0.1), np.sqrt(cumtrapz(mean_var, dx=0.1))

n_windows = 42
nbins = n_windows - 1

prot_name = "WT"
num_prot_atoms = 27086

qm_index2 = np.array([60, 53, 52], dtype=int) - 1 # MM Number to Python
qm_index = np.array([27261,26639,26638], dtype=int) - 1 # MM Number to Python

parm = AmberParm("../input/step3_pbcsetup.parm7")

pos_all = np.load('qm_pos.npy') # [ windows , frames , qm_atoms , coordinate ]

# got MM atom numbers from step5.00_equilibration.mdout
# grep "QMMM:     1" step5.00_equilibration.mdout -A 67 |awk '{print $3}' | tr '\n' ','

arr = np.array([13767,13768,13769,13770,13771,13772,13779,13780,13781,13782,13783,13784,13785,13786,13787,13788,
         13789,14154,14155,14156,14157,14158,14159,14160,14161,22499,26613,26614,26615,26616,26617,26618,
         26619,26620,26621,26622,26623,26624,26625,26626,26627,26628,26629,26630,26631,26632,26633,26634,
         26635,26636,26637,26638,26639,26640,26641,26642,26643,26644,26645], dtype=int) - 1

#water atoms
#27261,27262,27263,27267,27268,27269,27270,27271,27272]

# total number of non-water atoms minus 1
num_prot_atoms2 = 27145
resid = np.zeros(num_prot_atoms2, dtype=int)
for i in range(num_prot_atoms2):
    resid[i] = parm.atoms[i].residue.idx

resid = np.delete(resid,arr-1)
num_prot_res = resid[-1]

res_names = []
for i in np.unique(resid):
    if i < 1366:
        _res_id = i + 1
        res_names.append("Prot-" + parm.residues[i].name.capitalize() + "%d" % _res_id )
    elif 1366 <= i <= 1368:
        _res_id = i + 1
        res_names.append("MGs-" + parm.residues[i].name.capitalize() + "%d" % _res_id )
    elif i >= 1369 and i <= 1466:
        _res_id = i + 1
        res_names.append("RNA-" + parm.residues[i].name.capitalize() + "%d" % _res_id )
    elif i >= 1467 and i <= 1515:
        _res_id = i + 1
        res_names.append("DNA-" + parm.residues[i].name.capitalize() + "%d" % _res_id )

qmmm_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/qmmm_forces.npy" % i).swapaxes(1, 2) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    qmmm_forces.append(force[:, qm_index2])
qmmm_forces = np.array(qmmm_forces)

qmmm_prot_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/qmmm_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    qmmm_prot_forces.append(force.sum(axis=3))
qmmm_prot_forces = np.array(qmmm_prot_forces)

qmmm_res_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/qmmm_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    force2 = np.zeros((250, 3, 3, num_prot_res))
    for i in range(num_prot_res):
        force2[:, :, :, i] = force[:, :, :, resid == i].sum(axis=3)
    qmmm_res_forces.append(force2)
qmmm_res_forces = np.array(qmmm_res_forces)

qmmm_near_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/qmmm_near_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    qmmm_near_forces.append(force)
qmmm_near_forces = np.array(qmmm_near_forces)

qmmm_far_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/qmmm_far_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    qmmm_far_forces.append(force)
qmmm_far_forces = np.array(qmmm_far_forces)

gas_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/gas_forces.npy" % i).swapaxes(1, 2) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    gas_forces.append(force[:, qm_index2])
gas_forces = np.array(gas_forces)

prot_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/gas_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    prot_forces.append(force.sum(axis=3))
prot_forces = np.array(prot_forces)

res_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/gas_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    force2 = np.zeros((len(force), 3, 3, num_prot_res))
    for i in range(num_prot_res):
        force2[:, :, :, i] = force[:, :, :, resid == i].sum(axis=3)
    res_forces.append(force2)
res_forces = np.array(res_forces)

near_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/gas_near_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    near_forces.append(force)
near_forces = np.array(near_forces)

far_forces = []
for i in range(n_windows):
    force = -1 * np.load("../%02d/eda/gas_far_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    far_forces.append(force)
far_forces = np.array(far_forces)

vdw_prot_forces = []
for i in range(n_windows):
    force = np.load("../%02d/eda/lj_prot_forces_sum.npy" % i)
    vdw_prot_forces.append(force)
vdw_prot_forces = np.array(vdw_prot_forces)

vdw_res_forces = []
for i in range(n_windows):
    force = np.load("../%02d/eda/lj_prot_forces.npy" % i)
    force2 = np.zeros((len(force), 3, 3, num_prot_res))
    for i in range(num_prot_res):
        force2[:, :, :, i] = force[:, :, :, resid == i].sum(axis=3)
    vdw_res_forces.append(force2)
vdw_res_forces = np.array(vdw_res_forces)

vdw_near_forces = []
for i in range(n_windows):
    force = np.load("../%02d/eda/lj_near_forces.npy" % i)
    vdw_near_forces.append(force)
vdw_near_forces = np.array(vdw_near_forces)

pmf_geom, pmf_geom_std = get_pmf_geom(pos_all)
pmf_vdw, pmf_vdw_std = get_pmf(vdw_prot_forces + vdw_near_forces, pos_all)
pmf_total, pmf_total_std = get_pmf(vdw_prot_forces + vdw_near_forces + qmmm_forces, pos_all)

pmf_qmmm, pmf_qmmm_std = get_pmf(qmmm_forces, pos_all)
pmf_gas, pmf_gas_std = get_pmf(gas_forces, pos_all)
pmf_elec, pmf_elec_std = get_pmf(qmmm_forces - gas_forces, pos_all)
pmf_perm, pmf_perm_std = get_pmf(prot_forces + near_forces + far_forces, pos_all)
pmf_pol, pmf_pol_std = get_pmf(qmmm_forces - gas_forces - prot_forces - near_forces - far_forces, pos_all)

pmf_delta_qmmm, pmf_delta_qmmm_std = get_pmf(qmmm_prot_forces + qmmm_near_forces + qmmm_far_forces - prot_forces - near_forces - far_forces, pos_all)
pmf_dist, pmf_dist_std = get_pmf(qmmm_forces - gas_forces - qmmm_prot_forces - qmmm_near_forces - qmmm_far_forces, pos_all)

pmf_elec_res = np.zeros((num_prot_res, nbins))
pmf_elec_res_std = np.zeros((num_prot_res, nbins))
elec_res_forces = 0.5 * (qmmm_res_forces + res_forces)
for i in range(num_prot_res):
    pmf_elec_res[i], pmf_elec_res_std[i] = get_pmf(elec_res_forces[:, :, :, :, i], pos_all)
pmf_near_elec, pmf_near_elec_std = get_pmf(0.5 * (qmmm_near_forces + near_forces), pos_all)
pmf_far_elec, pmf_far_elec_std = get_pmf(0.5 * (qmmm_far_forces + far_forces), pos_all)

pmf_perm_res = np.zeros((num_prot_res, nbins))
pmf_perm_res_std = np.zeros((num_prot_res, nbins))
for i in range(num_prot_res):
    pmf_perm_res[i], pmf_perm_res_std[i] = get_pmf(res_forces[:, :, :, :, i], pos_all)
pmf_near_perm, pmf_near_perm_std = get_pmf(near_forces, pos_all)
pmf_far_perm, pmf_far_perm_std = get_pmf(far_forces, pos_all)

pmf_pol_res = np.zeros((num_prot_res, nbins))
pmf_pol_res_std = np.zeros((num_prot_res, nbins))
pol_res_forces = 0.5 * (qmmm_res_forces - res_forces)
for i in range(num_prot_res):
    pmf_pol_res[i], pmf_perm_res_std[i] = get_pmf(pol_res_forces[:, :, :, :, i], pos_all)
pmf_near_pol, pmf_near_pol_std = get_pmf(0.5 * (qmmm_near_forces - near_forces), pos_all)
pmf_far_pol, pmf_far_pol_std = get_pmf(0.5 * (qmmm_far_forces - far_forces), pos_all)

pmf_vdw_res = np.zeros((num_prot_res, nbins))
pmf_vdw_res_std = np.zeros((num_prot_res, nbins))
for i in range(num_prot_res):
    pmf_vdw_res[i], pmf_vdw_res_std[i] = get_pmf(vdw_res_forces[:, :, :, :, i], pos_all)
pmf_near_vdw, pmf_near_vdw_std = get_pmf(vdw_near_forces, pos_all)

pmf_total_res = np.zeros((num_prot_res, nbins))
pmf_total_res_std = np.zeros((num_prot_res, nbins))
total_res_forces = 0.5 * (qmmm_res_forces + res_forces) + vdw_res_forces
for i in range(num_prot_res):
    pmf_total_res[i], pmf_total_res_std[i] = get_pmf(total_res_forces[:, :, :, :, i], pos_all)
pmf_near_total, pmf_near_total_std = get_pmf(0.5 * (qmmm_near_forces + near_forces) + vdw_near_forces, pos_all)

pmf_total_res = np.zeros((num_prot_res, nbins))
pmf_total_res_std = np.zeros((num_prot_res, nbins))
total_res_forces = 0.5 * (qmmm_res_forces + res_forces)
for i in range(num_prot_res):
    pmf_total_res[i], pmf_total_res_std[i] = get_pmf(total_res_forces[:, :, :, :, i], pos_all)
pmf_near_total, pmf_near_total_std = get_pmf(0.5 * (qmmm_near_forces + near_forces), pos_all)

# Total-FEP-Decomposition

barrier_perm_res = np.zeros((num_prot_res, 2))
barrier_pol_res = np.zeros((num_prot_res, 2))
barrier_elec_res = np.zeros((num_prot_res, 2))
barrier_vdw_res = np.zeros((num_prot_res, 2))
barrier_total_res = np.zeros((num_prot_res, 2))

# Remove negative when correcting r1
maxbin=(pmf_total).argmax()
minbin=(pmf_total).argmin()

for i in range(num_prot_res):
    barrier_perm_res[i] = [pmf_perm_res[i, maxbin] - pmf_perm_res[i, minbin], pmf_perm_res_std[i, maxbin]]
    barrier_pol_res[i] = [pmf_pol_res[i, maxbin] - pmf_pol_res[i, minbin], pmf_pol_res_std[i, maxbin]]
    barrier_elec_res[i] = [pmf_elec_res[i, maxbin] - pmf_elec_res[i, minbin], pmf_elec_res_std[i, maxbin]]
    barrier_vdw_res[i] = [pmf_vdw_res[i, maxbin] - pmf_vdw_res[i, minbin], pmf_vdw_res_std[i, maxbin]]
    barrier_total_res[i] = [pmf_total_res[i, maxbin] - pmf_total_res[i, minbin], pmf_total_res_std[i, maxbin]]

barrier_near_perm = pmf_near_perm[maxbin] - pmf_near_perm[minbin]
barrier_near_perm_std = pmf_near_perm_std[maxbin]

barrier_far_perm = pmf_far_perm[maxbin] - [minbin]
barrier_far_perm_std = pmf_far_perm_std[maxbin]

barrier_near_pol = pmf_near_pol[maxbin] - pmf_near_pol[minbin]
barrier_near_pol_std = pmf_near_pol_std[maxbin]

barrier_far_pol = pmf_far_pol[maxbin] - pmf_far_pol[minbin]
barrier_far_pol_std = pmf_far_pol_std[maxbin]

barrier_near_elec = pmf_near_elec[maxbin] - pmf_near_elec[minbin]
barrier_near_elec_std = pmf_near_elec_std[maxbin]

barrier_far_elec = pmf_far_elec[maxbin] - pmf_far_elec[minbin]
barrier_far_elec_std = pmf_far_elec_std[maxbin]

barrier_near_vdw = pmf_near_vdw[maxbin] - pmf_near_vdw[minbin]
barrier_near_vdw_std = pmf_near_vdw_std[maxbin]

barrier_far_vdw = 0.0
barrier_far_vdw_std = 0.0

barrier_near_total = pmf_near_total[maxbin] - pmf_near_total[minbin]
barrier_near_total_std = pmf_near_total_std[maxbin]

barrier_total, barrier_total_std = pmf_total[maxbin] - pmf_total[minbin], pmf_total_std[maxbin]
barrier_vdw, barrier_vdw_std = pmf_vdw[maxbin] - pmf_vdw[minbin], pmf_vdw_std[maxbin]
barrier_gas, barrier_gas_std = pmf_gas[maxbin] - pmf_gas[minbin], pmf_gas_std[maxbin]
barrier_perm, barrier_perm_std = pmf_perm[maxbin] - pmf_perm[minbin], pmf_perm_std[maxbin]
barrier_pol, barrier_pol_std = pmf_pol[maxbin] - pmf_pol[minbin], pmf_pol_std[maxbin]
barrier_geom, barrier_geom_std = pmf_geom[maxbin] - pmf_geom[minbin], pmf_geom_std[maxbin]


# Visualization for Decomp.

mm_atom_forces = []
for i in range(n_windows):
    force1 = -1 * np.load("../%02d/eda/gas_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    force2 = -1 * np.load("../%02d/eda/qmmm_prot_forces.npy" % i) * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
    mm_atom_forces.append(0.5 * (force1 + force2))
mm_atom_forces = np.array(mm_atom_forces)

mm_barrier_pot = np.zeros((num_prot_atoms, 2))

for i in range(num_prot_atoms):
    pmf_mm_atom, pmf_mm_atom_std = get_pmf(mm_atom_forces[:, :, :, :, i], pos_all)
    barrier_mm_atom, barrier_mm_atom_std = pmf_mm_atom[maxbin] - pmf_mm_atom[minbin], pmf_mm_atom_std[maxbin]
    mm_barrier_pot[i] = barrier_mm_atom / parm.atoms[i].charge, barrier_mm_atom_std / parm.atoms[i].charge

np.save("mm_barrier_pot", mm_barrier_pot)
np.savetxt("mm_barrier_pot.dat", mm_barrier_pot, fmt="%5.2f")

table = [
    ["Gas", barrier_gas, barrier_gas_std],
    ["Perm.", barrier_perm, barrier_perm_std],
    ["Pol.", barrier_pol, barrier_pol_std],
    ["VdW", barrier_vdw, barrier_vdw_std],
    ["Geom.", barrier_geom, barrier_geom_std],
    ["Total", barrier_total + barrier_geom, barrier_total_std + barrier_geom_std]]

print(tabulate(table, tablefmt="latex_booktabs", floatfmt=".2f"))


multicol = pd.MultiIndex.from_tuples(
    [
        (prot_name, 'Perm.'), (prot_name, 'Perm. Std.'),
        (prot_name, 'Pol.'), (prot_name, 'Pol. Std.'),
        (prot_name, 'VdW'), (prot_name, 'VdW Std.'),
        (prot_name, 'Total'), (prot_name, 'Total Std.'),
     ])

df = pd.DataFrame(data=np.array(
    [
        barrier_perm_res[:, 0], barrier_perm_res[:, 1],
        barrier_pol_res[:, 0], barrier_pol_res[:, 1],
        barrier_vdw_res[:, 0], barrier_vdw_res[:, 1],
        barrier_total_res[:, 0], barrier_total_res[:, 1],
    ]).T,
    columns=multicol)

df2 = pd.DataFrame(data=np.array([
    [
        barrier_near_perm, barrier_near_perm_std,
        barrier_near_pol, barrier_near_pol_std,
        barrier_near_vdw, barrier_near_vdw_std,
        barrier_near_total, barrier_near_total_std,
    ],
    [
        barrier_far_perm, barrier_far_perm_std,
        barrier_far_pol, barrier_far_pol_std,
        barrier_far_vdw, barrier_far_vdw_std,
        barrier_far_elec, barrier_far_elec_std,
    ]]), columns=multicol)

df = df.append(df2)
df.index = res_names + ["Near Solv.", "Far Solv."]
df.to_pickle("barrier_res_table-new3.pkl")

df3 = df[abs(df[(prot_name, 'Total')]) > 0.3].sort_values(by=(prot_name, 'Total'))
#print(tabulate(df3, tablefmt="latex_booktabs", floatfmt=".2f"))
