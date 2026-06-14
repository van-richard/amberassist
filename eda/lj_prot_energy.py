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

qm_index = np.array([13767,13768,13769,13770,13771,13772,13779,13780,13781,13782,13783,13784,13785,13786,13787,13788,
         13789,14154,14155,14156,14157,14158,14159,14160,14161,22499,26613,26614,26615,26616,26617,26618,
         26619,26620,26621,26622,26623,26624,26625,26626,26627,26628,26629,26630,26631,26632,26633,26634,
         26635,26636,26637,26638,26639,26640,26641,26642,26643,26644,26645,27261,27262,27263,27267,27268,27269,27270,27271,27272])

fin = (f"qmhub/qmmm.inp_{i:04d}" for i in range(frames))

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
            if dij[qm_index[j], ii] == 0.0:
                print("Same Atom!")
            elif dij[qm_index[j], ii] < 10.0:
                epsilon = np.sqrt(parm.atoms[ii].epsilon * parm.atoms[qm_index[j]].epsilon)
                acoeff = epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index[j]].rmin)**12
                bcoeff = 2 * epsilon * (parm.atoms[ii].rmin + parm.atoms[qm_index[j]].rmin)**6
                ene = acoeff * d_inv[qm_index[j], ii]**12 - bcoeff * d_inv[qm_index[j], ii]**6
                energy[j, ii] = ene

    prot_energy.append(energy)
    print(f, ' finished at ', os.system('date'))

np.save("lj_prot_energy", prot_energy)
