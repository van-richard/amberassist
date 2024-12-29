import sys
from glob import glob
from pathlib import Path

import numpy as np

from qmhub import QMMM
from qmhub.units import CODATA08_BOHR_TO_A
import os

total_frames=1
qm_atoms=73
mm_atoms=800

cwdir = os.getcwd()

idx = sys.argv[1]

fin = glob(f"qmhub/qmmm.inp_{idx}")[0]
tmpdir=f"dft_forces/{idx}"
os.makedirs(tmpdir, exist_ok=True)
print(fin, tmpdir)

qmmm = QMMM(mode="text", driver="sander", cwd=Path(tmpdir))
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

qm_coord = np.zeros((total_frames, qm_atoms, 3)) 
mm_coord = np.zeros((total_frames, mm_atoms, 3)) 
mm_charge = np.zeros((total_frames, mm_atoms))

qmmm.load_system(fin)
qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
qmmm.add_engine("qchem", options={
    "method": "b3lyp",
    "basis": "6-31gd",
    "scf_convergence": "6",},)

index = np.argsort(abs(qmmm.engine.mm_charges))[-mm_atoms:]
mm_coord[0] = qmmm.engine.mm_positions.T[index]
mm_charge[0] = qmmm.engine.mm_charges[index]
qm_coord[0] = qmmm.engine.qm_positions.T
print('Finished: Frame ' + str(fin))

np.save(f"{tmpdir}/qm_coord", np.array(qm_coord, dtype="float32"))
np.save(f"{tmpdir}/mm_coord", np.array(mm_coord, dtype="float32"))
np.save(f"{tmpdir}/mm_charge", np.array(mm_charge, dtype="float32"))
