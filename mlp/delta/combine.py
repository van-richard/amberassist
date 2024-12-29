import numpy as np
import sys

mm_charge = np.empty([36288, 773])
mm_coord = np.empty([36288, 773, 3])
qm_coord = np.empty([36288, 68, 3])

for i in range(42):
    mm_charges = np.load('../%02d/mm_charge.npy' %i)
    mm_charge_slice = mm_charges
    mm_charge[i*864:(i+1)*864,:] = mm_charge_slice
    
    mm_coords = np.load('../%02d/mm_coord.npy' %i)
    mm_coord_slice = mm_coords
    mm_coord[i*864:(i+1)*864,:] = mm_coord_slice
    
    qm_coords = np.load('../%02d/qm_coord.npy' %i)
    qm_coord_slice = qm_coords
    qm_coord[i*864:(i+1)*864,:] = qm_coord_slice
    
np.save("mm_charge", mm_charge)
np.save("mm_coord", mm_coord)
np.save("qm_coord", qm_coord)
