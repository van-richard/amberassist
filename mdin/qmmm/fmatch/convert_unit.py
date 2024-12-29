import numpy as np
import os

qmcoord = np.load('dft_forces/qm_coord.npy')
mmcoord = np.load('dft_forces/mm_coord.npy')
mmcharge = np.load('dft_forces/mm_charge.npy')

E = np.load("training_set/energy.npy") * 27.2114
F = -np.load("training_set/qm_grad.npy") * 27.2114 / 0.529177249
esp = np.load("training_set/mm_esp.npy") * 27.2114
esp_grad = np.load("training_set/mm_esp_grad.npy") * 27.2114 / 0.529177249

#atom_types = np.array(np.loadtxt("../input/qm_elem.txt"), dtype="int64")
#atom_types = np.array([np.unique(atom_types).tolist().index(i) for i in atom_types])
#np.save("qm_type.npy", atom_types)

iters=50
out='test'
os.makedirs(out, exist_ok=True) 
np.save(f"{out}/qm_coord", qmcoord[::iters])
np.save(f"{out}/mm_coord", mmcoord[::iters])
np.save(f"{out}/mm_charge", mmcharge[::iters])
np.save(f"{out}/energy", E[::iters])
np.save(f"{out}/qm_force", F[::iters])
np.save(f"{out}/mm_esp", esp[::iters])
np.save(f"{out}/mm_esp_grad", esp_grad[::iters])

