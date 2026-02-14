import os
import sys
import torch
from torchmdnet.module import LNNP

# checkpoint = sys.argv[1]
qm_atoms=73
mm_atoms=800

inp = sys.argv[1]
model = torch.jit.load(inp)
#model = torch.jit.load('output/model-cpu.pt')

import numpy as np
qm_coord = torch.from_numpy(np.array(np.load("dft_forces/qm_coord.npy"), dtype="float32"))
atom_types = torch.from_numpy(np.array(np.load("qm_type.npy"), dtype="int64"))
atom_types = atom_types.repeat(len(qm_coord), 1)

charge_coord = torch.from_numpy(np.array(np.load("dft_forces/mm_coord.npy"), dtype="float32"))
charges = torch.from_numpy(np.array(np.load("dft_forces/mm_charge.npy"), dtype="float32"))

ene_pred = np.zeros(len(qm_coord))
grad_pred = np.zeros((len(qm_coord), qm_atoms, 3))
mm_esp_pred = np.zeros((len(qm_coord), mm_atoms))
mm_esp_grad_pred = np.zeros((len(qm_coord), mm_atoms, 3))
for i, (at, c, qc, q) in enumerate(zip(atom_types, qm_coord, charge_coord, charges)):
    ene, grad, esp, esp_grad = model(at, c, qc, q)
    ene_pred[i] = ene.detach().numpy()
    grad_pred[i] = grad.detach().numpy()
    mm_esp_pred[i] = esp.detach().numpy()
    mm_esp_grad_pred[i] = esp_grad.detach().numpy()
#ene_pred = ene_pred - ene_pred.mean()

output = sys.argv[2]
os.makedirs(output, exist_ok=True)
np.save("%s/ene_pred" % output, ene_pred)
np.save("%s/grad_pred" % output, grad_pred)
np.save("%s/mm_esp_pred" % output, mm_esp_pred)
np.save("%s/mm_esp_grad_pred" % output, mm_esp_grad_pred)
