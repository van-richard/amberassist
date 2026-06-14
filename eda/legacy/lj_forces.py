#!/usr/bin/env python
from parmed.amber import NetCDFTraj, AmberParm
import numpy as np
import math

def dij(x1, x2):
    v = x2 - x1
    r = np.linalg.norm(v)
    return r

def lennard_jones_energy(x1, x2, epsilon, sigma):
    v = x2 - x1
    r = np.linalg.norm(v)
    e = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    f = ((4 * epsilon) / r) * (-12 * (sigma / r)**12 + 6 * (sigma / r)**6) * (v / r)
    return e, f

parm = AmberParm('../input/step3_pbcsetup.parm7')

nonbonded_exclusion = [set() for _ in parm.atoms]

for bond in parm.bonds:
    nonbonded_exclusion[bond.atom1.idx].add(bond.atom2.idx)
    nonbonded_exclusion[bond.atom2.idx].add(bond.atom1.idx)

for angle in parm.angles:
    nonbonded_exclusion[angle.atom1.idx].add(angle.atom3.idx)
    nonbonded_exclusion[angle.atom3.idx].add(angle.atom1.idx)

for dihedral in parm.dihedrals:
    nonbonded_exclusion[dihedral.atom1.idx].add(dihedral.atom4.idx)
    nonbonded_exclusion[dihedral.atom4.idx].add(dihedral.atom1.idx)    

pos_all = []
pos = []
traj = NetCDFTraj.open_old("step7_reprocessing.nc")
pos.append(traj.coordinates)
traj.close()
pos_all.append(np.concatenate(pos)[0:500:2])

num_prot_atoms = 27086

qm_atoms = np.array([27261,26639,26638], dtype=int) - 1 # CV atoms

arr = np.array([13767,13768,13769,13770,13771,13772,13779,13780,13781,13782,13783,13784,13785,13786,13787,13788,
         13789,14154,14155,14156,14157,14158,14159,14160,14161,22499,26613,26614,26615,26616,26617,26618,
         26619,26620,26621,26622,26623,26624,26625,26626,26627,26628,26629,26630,26631,26632,26633,26634,
         26635,26636,26637,26638,26639,26640,26641,26642,26643,26644,26645,
         27261,27262,27263,27267,27268,27269,27270,27271,27272], dtype=int) -1 # all QM atoms

prot_forces = []
near_forces = []

for frame in range(len(pos_all[0])):
    forces = np.zeros((len(qm_atoms), 3, len(parm.atoms)))
    for i, atom in enumerate(qm_atoms):
        atom1 = pos_all[0][frame][atom]
        for j, atoms in enumerate(pos_all[0][frame]):   
            atom2 = pos_all[0][frame][j]
            if atom != j and dij(atom1, atom2) < 10 and not j in nonbonded_exclusion[j] and not j in arr:
                epsilon = np.sqrt(parm.atoms[atom].epsilon * parm.atoms[j].epsilon)
                sigma = (parm.atoms[atom].sigma + parm.atoms[j].sigma) / 2.0
                e, f = lennard_jones_energy(atom1, atom2, epsilon, sigma)
                forces[i, :, j] = f
    prot_forces.append(forces[:, :, :num_prot_atoms])
    near_forces.append(forces[:, :, num_prot_atoms:].sum(axis=2))

np.save('lj_prot_forces.npy', prot_forces)
np.save('lj_near_forces.npy', near_forces)
