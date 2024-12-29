import numpy as np

qmcoord = np.load('files/qm_coord.npy')[::21]
mmcoord = np.load('files/mm_coord.npy')[::21]
mmcharge = np.load('files/mm_charge.npy')[::21]

np.save('qm_coord', qmcoord)
np.save('mm_coord', mmcoord)
np.save('mm_charge', mmcharge)

E = np.load("files/energy.npy")[::21] * 27.2114
F = -np.load("files/qm_grad.npy")[::21] * 27.2114 / 0.529177249
esp = np.load("files/mm_esp.npy")[::21] * 27.2114
esp_grad = np.load("files/mm_esp_grad.npy")[::21] * 27.2114 / 0.529177249
#esp_grad = -np.load("mm_efield.npy") * 27.2114 / 0.529177249

np.save("energy", E)
np.save("qm_force", F)
np.save("mm_esp", esp)
np.save("mm_esp_grad", esp_grad)

atom_types = np.array(np.loadtxt("files/qm_elem.txt"), dtype="int64")
atom_types = np.array([np.unique(atom_types).tolist().index(i) for i in atom_types])
np.save("qm_type.npy", atom_types)

# print(qmcoord.shape)
# print(mmcoord.shape)
# print(mmcharge.shape)
# print(E.shape, F.shape, esp.shape, esp_grad.shape)
# print(atom_types.shape)
