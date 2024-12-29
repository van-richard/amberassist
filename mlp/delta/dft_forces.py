#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np

from qmhub import QMMM
from qmhub.units import CODATA08_BOHR_TO_A

fin = (f"test/qmmm.inp_{i:04d}" for i in range(864))

qmmm = QMMM(mode="text", driver="sander", cwd=Path("/dev/shm/__WIN__qmhub"))
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

qm_coord = np.zeros((864, 68, 3)) 
mm_coord = np.zeros((864, 773, 3)) 
mm_charge = np.zeros((864, 773))

for i, f in enumerate(fin):
	if i == 0:
		qmmm.load_system(f)
		qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
		qmmm.add_engine(
			"qchem",
			options={
				"method": "b3lyp",
				"basis": "6-31g*",
				"scf_convergence": "9",
				},
			)
	else:
		qmmm.io.load_system(f, system=qmmm.system)

	index = np.argsort(abs(qmmm.engine.mm_charges))[-773:]
	mm_coord[i] = qmmm.engine.mm_positions.T[index]
	mm_charge[i] = qmmm.engine.mm_charges[index]
	qm_coord[i] = qmmm.engine.qm_positions.T

np.save("qm_coord", np.array(qm_coord, dtype="float32"))
np.save("mm_coord", np.array(mm_coord, dtype="float32"))
np.save("mm_charge", np.array(mm_charge, dtype="float32"))
