#!/usr/bin/env python
import sys
import os
from pathlib import Path

import numpy as np

from qmhub import QMMM
from qmhub.units import CODATA08_BOHR_TO_A

pstep = 0
frames = 500
num_qm_atoms = 73
num_prot_atoms = 27086

# These files are expected to be visible from a window directory. They are
# generated from the packed qmhub.squashfs outside this directory; this script
# does not unpack or regenerate that reference data.
input_dir = Path("qmhub")
input_files = [input_dir / f"qmmm.inp_{i:04d}" for i in range(pstep, frames, 2)]
missing_inputs = [path for path in input_files if not path.is_file()]
if missing_inputs:
    raise FileNotFoundError(
        "Missing QMHub input files; run from a window directory with qmhub/"
        f" visible. First missing file: {missing_inputs[0]}"
    )
fin = (str(path) for path in input_files)

output_dir = Path(os.environ.get("EDA_OUTPUT_DIR", "eda"))
output_dir.mkdir(parents=True, exist_ok=True)
work_dir = Path(os.environ.get("EDA_WORK_DIR", "eda/gas_work"))

qm_index = np.array([60, 53, 52], dtype=int) - 1

qmmm = QMMM(mode="text", driver="sander", cwd=work_dir)
qmmm.io.cwd.mkdir(exist_ok=True)
qmmm.setup_simulation()

energy = []
forces = []
prot_forces = []
near_forces = []
far_forces = []
mm_esp_all = []

for i, f in enumerate(fin):
    if i == 0:
        qmmm.load_system(f)
        qmmm.build_model(switching_type='lrec', cutoff=10., swdist=None, pbc=True)
        qmmm.add_engine(
            "qchem",
            options={
                "method": "b3lyp",
                "basis": "6-31g*",
                "scf_convergence": "9",
            },
        )
    else:
        qmmm.io.load_system(f, system=qmmm.system)

    atom_charges = np.copy(qmmm.system.atoms.charges)
    qm_total_esp = np.copy(qmmm.model.elec.full.qm_total_esp)
    qmmm.system.atoms.charges[:] = 0.0

    mm_esp_grads = np.zeros((len(qm_index), 3, len(qmmm.system.atoms)))
    qm_esp_charges_grads = np.zeros((len(qm_index), 3, len(qmmm.system.qm.atoms)))
    # mm_esp2_grads = np.zeros((len(qm_index), 3, len(qmmm.system.qm.atoms), qmmm.engine.mm_esp.shape[1]))
    for ii, k in enumerate(qm_index):
        for j in range(3):
            mm_esp = qmmm.model.engine.mm_esp
            qm_esp_charges = qmmm.model.engine.qm_esp_charges

            qmmm.system.atoms.positions[j, k:k+1] += 0.0001
            # mm_esp_pos = np.copy(mm_esp[0])
            mm_esp_pos = np.zeros(len(qmmm.system.atoms))
            mm_esp_pos[qmmm.model.elec.near_field.near_field_mask] = np.copy(mm_esp[0])
            qm_esp_charges_pos = np.copy(qmmm.model.engine.qm_esp_charges)
            # mm_esp2_pos = np.copy(qmmm.model.engine.qm_esp_charges[:, None] * qmmm.model.elec.near_field.qmmm_coulomb_tensor)
            qmmm.system.atoms.positions[j, k:k+1] -= 0.0002
            # mm_esp_neg = np.copy(mm_esp[0])
            mm_esp_neg = np.zeros(len(qmmm.system.atoms))
            mm_esp_neg[qmmm.model.elec.near_field.near_field_mask] = np.copy(mm_esp[0])
            qm_esp_charges_neg = np.copy(qmmm.model.engine.qm_esp_charges)
            # mm_esp2_neg = np.copy(qmmm.model.engine.qm_esp_charges[:, None] * qmmm.model.elec.near_field.qmmm_coulomb_tensor)
            qmmm.system.atoms.positions[j, k:k+1] += 0.0001

            mm_esp_grads[ii, j] = (mm_esp_pos - mm_esp_neg) / 0.0002
            qm_esp_charges_grads[ii, j] = (qm_esp_charges_pos - qm_esp_charges_neg) / 0.0002
            # mm_esp2_grads[ii, j] = (mm_esp2_pos - mm_esp2_neg) / 0.0002

    energy.append(np.copy(qmmm.simulation.energy))
    forces.append(np.copy(qmmm.simulation.energy_gradient[:, qmmm.system.qm_index]))

    # mm_esp_grads_all.append(mm_esp_grads)
    # qm_esp_charges_grads_all.append(qm_esp_charges_grads)
    # mm_esp2_grads_all.append(mm_esp2_grads)
    # mm_esp_all.append(np.copy(qmmm.model.engine.mm_esp[0]))
    # qm_esp_charges_all.append(np.copy(qmmm.model.engine.qm_esp_charges))

    mask = np.copy(qmmm.model.elec.near_field.near_field_mask)
    mask[num_qm_atoms:num_qm_atoms+num_prot_atoms] = True

    w = np.zeros((len(mask)))
    w[qmmm.model.elec.near_field.near_field_mask] = qmmm.model.elec.near_field.scaling_factor
    w = w[mask]

    w_grad = np.zeros((len(qm_index), 3, len(mask)))
    w_grad[:, :, qmmm.model.elec.near_field.near_field_mask] = -qmmm.model.elec.near_field.scaling_factor_gradient[:, qm_index].swapaxes(0, 1)
    w_grad = w_grad[:, :, mask]

    charges = atom_charges[mask]
    d_inv = qmmm.model.elec.dij_inverse[:, mask]
    d_inv_grad = -qmmm.model.elec.dij_inverse_gradient[:, qm_index].swapaxes(0, 1)[:, :, mask]

    # mm_esp_grad = np.zeros((len(qm_index), 3, len(mask)))
    # mm_esp_grad[:, :, qmmm.model.elec.near_field.near_field_mask] = mm_esp_grads
    mm_esp_grad = mm_esp_grads[:, :, mask]

    qm_esp_charges = np.copy(qmmm.model.engine.qm_esp_charges)
    qm_esp_charges_grad = qm_esp_charges_grads

    phi_a = np.zeros((len(mask)))
    phi_a[qmmm.model.elec.near_field.near_field_mask] = np.copy(qmmm.model.engine.mm_esp[0])
    phi_a = phi_a[mask]
    phi_b = qm_esp_charges @ d_inv

    mm_esp = phi_a * w + phi_b * (1 - w)
    mm_esp_all.append(mm_esp[:num_prot_atoms])

    term1 = mm_esp_grad * charges * w
    term2 = (qm_esp_charges_grad @ d_inv + d_inv_grad * qm_esp_charges[qm_index, None, None]) * charges * (1 - w)
    term3 = w_grad * charges * (phi_a - phi_b)
    force = (term1 + term2 + term3) * CODATA08_BOHR_TO_A**2

    near = (qm_esp_charges_grad @ d_inv + d_inv_grad * qm_esp_charges[qm_index, None, None]) @ charges * CODATA08_BOHR_TO_A**2
    total = (qm_esp_charges_grad @ qm_total_esp[0] + (qm_total_esp[1:, qm_index] * qm_esp_charges[qm_index]).T) * CODATA08_BOHR_TO_A**2
    far = total - near

    prot_forces.append(force[:, :, :num_prot_atoms])
    near_forces.append(force[:, :, num_prot_atoms:].sum(axis=2))
    far_forces.append(far)
    print(f, ' finished at ', os.system('date'))

np.save(output_dir / "gas_energy.npy", energy)
np.save(output_dir / "gas_forces.npy", forces)
np.save(output_dir / "gas_prot_forces.npy", prot_forces)
np.save(output_dir / "gas_near_forces.npy", near_forces)
np.save(output_dir / "gas_far_forces.npy", far_forces)
np.save(output_dir / "gas_mm_esp.npy", mm_esp_all)
