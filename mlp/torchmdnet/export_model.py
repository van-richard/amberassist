import sys
import torch
from torchmdnet.module import LNNP

#checkpoint = sys.argv[1]

#model = LNNP.load_from_checkpoint(checkpoint)
#script = model.to_torchscript()
#torch.jit.save(script, "model.pt")
model = torch.jit.load('output4/model.pt')

import numpy as np
qm_coord = torch.from_numpy(np.array(np.load("qm_coord.npy"), dtype="float32"))
atom_types = torch.from_numpy(np.array(np.load("qm_type.npy"), dtype="int64"))
atom_types = atom_types.repeat(len(qm_coord), 1)

charge_coord = torch.from_numpy(np.array(np.load("mm_coord.npy"), dtype="float32"))
charges = torch.from_numpy(np.array(np.load("mm_charge.npy"), dtype="float32"))

ene_pred = np.zeros(len(qm_coord))
grad_pred = np.zeros((2000, 24, 3))
mm_esp_pred = np.zeros((2000, 850))
mm_esp_grad_pred = np.zeros((2000, 850, 3))
for i, (at, c, qc, q) in enumerate(zip(atom_types, qm_coord, charge_coord, charges)):
    ene, grad, esp, esp_grad = model(at, c, qc, q)
    ene_pred[i] = ene.detach().numpy()
    grad_pred[i] = grad.detach().numpy()
    mm_esp_pred[i] = esp.detach().numpy()
    mm_esp_grad_pred[i] = esp_grad.detach().numpy()
ene_pred = ene_pred - ene_pred.mean()

np.save("ene_pred", ene_pred)
np.save("grad_pred", grad_pred)
np.save("mm_esp_pred", mm_esp_pred)
np.save("mm_esp_grad_pred", mm_esp_grad_pred)
