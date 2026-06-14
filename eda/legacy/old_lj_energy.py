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

qm_index = np.arange(num_qm_atoms)
qm_index2 = qm_index + num_prot_atoms

parm = AmberParm("../input/step3_pbcsetup.parm7")

qmmm = QMMM(mode="text", driver="sander", cwd=Path("lj_energy"))
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

prot_energy = []

for i, f in enumerate(fin):
    if i == 0:
        qmmm.load_system(f)
        qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
    else:
        qmmm.io.load_system(f, system=qmmm.system)

    energy = np.zeros((len(qm_index), num_prot_atoms))

    rij = qmmm.model.elec.rij
    dij = qmmm.model.elec.dij
    d_inv = qmmm.model.elec.dij_inverse

    for ii in range(num_prot_atoms):
        for j in range(len(qm_index)):
            if dij[qm_index[j], ii+num_qm_atoms] < 10.0:
                epsilon = np.sqrt(parm.atoms[ii].epsilon * parm.atoms[qm_index2[j]].epsilon)
                acoeff = epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index2[j]].rmin)**12
                bcoeff = 2 * epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index2[j]].rmin)**6
                ene = acoeff * d_inv[qm_index[j], ii+num_qm_atoms]**12 - bcoeff * d_inv[qm_index[j], ii+num_qm_atoms]**6
                energy[j, ii] = ene

    prot_energy.append(energy)
    print(f, ' finished at ', os.system('date'))

np.save("lj_prot_energy", prot_energy)
