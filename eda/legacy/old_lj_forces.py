#!/usr/bin/env python
import sys
import os
from pathlib import Path

import numpy as np

from parmed.amber import AmberParm
from qmhub import QMMM

frames = 10
num_qm_atoms = 73
num_prot_atoms = 27086

fin = (f"qmhub/qmmm.inp_{i:04d}" for i in range(frames))

qm_index = np.array([52, 53, 53, 60])
qm_index2 = qm_index + num_prot_atoms
#/num_prot_atoms1 = qm_index2[3]

parm = AmberParm("../input/step3_pbcsetup.parm7")

qmmm = QMMM(mode="text", driver="sander", cwd=Path("lj_forces"))
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

prot_forces = []
near_forces = []

for i, f in enumerate(fin):
    if i == 0:
        qmmm.load_system(f)
        qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
    else:
        qmmm.io.load_system(f, system=qmmm.system)

    forces = np.zeros((len(qm_index), 3, len(qmmm.system.atoms)))

    mask = np.copy(qmmm.model.elec.near_field.near_field_mask)

    rij = qmmm.model.elec.rij
    dij = qmmm.model.elec.dij
    d_inv = qmmm.model.elec.dij_inverse

    for i in range(len(qmmm.system.atoms)):
        print(i)
        if mask[i]:
            for j in range(len(qm_index)):
                print(j)
                if dij[qm_index[j], i] < 10.0:
                    if i < num_prot_atoms:
                        ii = i - num_qm_atoms
                    elif i > num_prot_atoms:
                        ii = i
                    epsilon = np.sqrt(parm.atoms[ii].epsilon * parm.atoms[qm_index2[j]].epsilon)
                    acoeff = epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index2[j]].rmin)**12
                    bcoeff = 2 * epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index2[j]].rmin)**6
                    grad = -12 * acoeff * d_inv[qm_index[j], i]**13 + 6 * bcoeff * d_inv[qm_index[j], i]**7
                    grad *= rij[:, qm_index[j], i] * d_inv[qm_index[j], i]
                    forces[j, :, i] = grad
    np.save('forces.npy', forces)
    prot_forces.append(forces[:, :, num_qm_atoms:num_qm_atoms+num_prot_atoms])
    near_forces.append(forces[:, :, num_qm_atoms+num_prot_atoms:].sum(axis=2))
    print(f, ' finished at ', os.system('date'))

np.save("lj_prot_forces", prot_forces)
np.save("lj_near_forces", near_forces)
