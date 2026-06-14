import numpy as np

f = np.load('lj_prot_forces.npy')
force = f.sum(axis=3)

np.save('lj_prot_forces_sum.npy', force)
