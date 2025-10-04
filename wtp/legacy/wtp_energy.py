#!/usr/bin/env python
import sys
import os
from pathlib import Path

import numpy as np

from qmhub import QMMM
from qmhub.units import CODATA08_BOHR_TO_A

# filename (qmmm.inp_????)
fin = sys.argv[1]

qm_functional = "b3lyp"
qm_basis_set   = "6-31+gd"

# atom index of d1 - d2, use same atoms as cv.rst
# 0-index from QM region in qmmm.inp_????
qm_index = np.array([59, 52, 51])

qmmm = QMMM(mode="text", driver="sander", cwd=Path("./"))
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

energy = []
forces = []

qmmm.load_system(fin)
qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
qmmm.add_engine(
    "qchem",
    options={
        "method": f"{qm_functional}",
        "basis": f"{qm_basis_set}",
        "scf_convergence": "9",
    },
)

energy.append(np.copy(qmmm.simulation.energy))
forces.append(np.copy(qmmm.simulation.energy_gradient[:, qmmm.system.qm_index]))

#print(f"finished: {fin}\t{os.system('date')}")

np.save(f"qmmm_{qm_functional}_{qm_basis_set}_energy", energy)
np.save(f"qmmm_{qm_functional}_{qm_basis_set}_forces", forces)
