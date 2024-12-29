import numpy as np
import sys

energy1 = np.empty([36288])
mm_efield1 = np.empty([36288, 773, 3])
mm_esp1 = np.empty([36288, 773])
qm_grad1 = np.empty([36288, 68, 3])

for i in range(0,41):
    energy1s = np.load('../%02d/energy1.npy' %i)
    energy1_slice = energy1s
    energy1[i*864:(i+1)*864] = energy1_slice
    
    mm_efield1s = np.load('../%02d/mm_efield1.npy' %i)
    mm_efield1_slice = mm_efield1s
    mm_efield1[i*864:(i+1)*864,:] = mm_efield1_slice
    
    mm_esp1s = np.load('../%02d/mm_esp1.npy' %i)
    mm_esp1_slice = mm_esp1s
    mm_esp1[i*864:(i+1)*864,:] = mm_esp1_slice

    qm_grad1s = np.load('../%02d/qm_grad1.npy' %i)
    qm_grad1_slice = qm_grad1s
    qm_grad1[i*864:(i+1)*864,:] = qm_grad1_slice
    
np.save("energy1", energy1)
np.save("mm_efield1", mm_efield1)
np.save("mm_esp1", mm_esp1)
np.save("qm_grad1", qm_grad1)
