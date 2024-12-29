import numpy as np
import sys

energy_sqm = np.empty([36288])
mm_efield_sqm = np.empty([36288, 773, 3])
mm_esp_sqm = np.empty([36288, 773])
qm_grad_sqm = np.empty([36288, 68, 3])

for i in range(42):
    energy_sqms = np.load('../%02d/energy_sqm.npy' %i)
    energy_sqmslice = energy_sqms
    energy_sqm[i*864:(i+1)*864] = energy_sqmslice
    
    mm_efield_sqms = np.load('../%02d/mm_efield_sqm.npy' %i)
    mm_efield_sqmslice = mm_efield_sqms
    mm_efield_sqm[i*864:(i+1)*864,:] = mm_efield_sqmslice
    
    mm_esp_sqms = np.load('../%02d/mm_esp_sqm.npy' %i)
    mm_esp_sqmslice = mm_esp_sqms
    mm_esp_sqm[i*864:(i+1)*864,:] = mm_esp_sqmslice

    qm_grad_sqms = np.load('../%02d/qm_grad_sqm.npy' %i)
    qm_grad_sqmslice = qm_grad_sqms
    qm_grad_sqm[i*864:(i+1)*864,:] = qm_grad_sqmslice
    
np.save("energy_sqm", energy_sqm)
np.save("mm_efield_sqm", mm_efield_sqm)
np.save("mm_esp_sqm", mm_esp_sqm)
np.save("qm_grad_sqm", qm_grad_sqm)
