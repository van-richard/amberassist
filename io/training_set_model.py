import os
import sys
import torch
from torchmdnet.module import LNNP

qm_atoms=73
mm_atoms=800
n_frames=6300

results="pred"
os.makedirs(results, exist_ok=True)


#checkpoint = sys.argv[1]

#model = LNNP.load_from_checkpoint(checkpoint)
#script = model.to_torchscript()
#torch.jit.save(script, "model.pt")
model = torch.jit.load('../input/model0729-cpu.pt')

import numpy as np
qm_coord = torch.from_numpy(np.array(np.load("qm_coord.npy"), dtype="float32"))
charge_coord = torch.from_numpy(np.array(np.load("mm_coord.npy"), dtype="float32"))
charges = torch.from_numpy(np.array(np.load("mm_charge.npy"), dtype="float32"))
atom_types = torch.from_numpy(np.array(np.load("qm_elem.npy"), dtype="int64"))
atom_types = atom_types.repeat(len(qm_coord), 1)

ene_pred = np.zeros(len(qm_coord))
grad_pred = np.zeros((n_frames, qm_atoms, 3))
mm_esp_pred = np.zeros((n_frames, mm_atoms))
mm_esp_grad_pred = np.zeros((n_frames, mm_atoms, 3))
for i, (at, c, qc, q) in enumerate(zip(atom_types, qm_coord, charge_coord, charges)):
    ene, grad, esp, esp_grad = model(at, c, qc, q)
    ene_pred[i] = ene.detach().numpy()
    grad_pred[i] = grad.detach().numpy()
    mm_esp_pred[i] = esp.detach().numpy()
    mm_esp_grad_pred[i] = esp_grad.detach().numpy()
ene_pred = ene_pred - ene_pred.mean()

np.save("predenergy", ene_pred)
np.save("predqm_grad", grad_pred)
np.save("predmm_esp", mm_esp_pred)
np.save("predmm_esp_grad", mm_esp_grad_pred)
