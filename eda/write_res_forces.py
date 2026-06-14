#!/usr/bin/env python
from pathlib import Path

import numpy as np
from parmed.amber import AmberParm
from qmhub.units import AMBER_BOHR_TO_A, AMBER_HARTREE_TO_KCAL

N_WINDOWS = 42
N_FRAMES = 250
NUM_PROT_ATOMS = 27086
PARM_PATH = Path("../input/step3_pbcsetup.parm7")

QM_INDEX_EDA = np.array([60, 53, 52], dtype=int) - 1

# MM atom numbers from step5.00_equilibration.mdout:
# grep "QMMM:     1" step5.00_equilibration.mdout -A 67 | awk '{print $3}' | tr '\n' ','
EXCLUDED_QM_ATOMS = np.array(
    [
        13767, 13768, 13769, 13770, 13771, 13772, 13779, 13780,
        13781, 13782, 13783, 13784, 13785, 13786, 13787, 13788,
        13789, 14154, 14155, 14156, 14157, 14158, 14159, 14160,
        14161, 22499, 26613, 26614, 26615, 26616, 26617, 26618,
        26619, 26620, 26621, 26622, 26623, 26624, 26625, 26626,
        26627, 26628, 26629, 26630, 26631, 26632, 26633, 26634,
        26635, 26636, 26637, 26638, 26639, 26640, 26641, 26642,
        26643, 26644, 26645,
    ],
    dtype=int,
) - 1


def window_eda_path(window, filename):
    return Path(f"../{window:02d}/eda") / filename


def load_window_array(window, filename):
    path = window_eda_path(window, filename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing EDA input: {path}")
    return np.load(path)


def save_array(path, array):
    np.save(path, array)
    print(f"saved {path}")


def build_residue_metadata(parm):
    # Matches the notebook's non-water atom range and QM atom removal.
    num_prot_atoms2 = 27146 - 1
    resid = np.zeros(num_prot_atoms2, dtype=int)
    for atom_idx in range(num_prot_atoms2):
        resid[atom_idx] = parm.atoms[atom_idx].residue.idx

    resid = np.delete(resid, EXCLUDED_QM_ATOMS)
    num_prot_res = resid[-1]

    res_names = []
    for res_idx in np.unique(resid):
        if res_idx <= 1364:
            res_id = res_idx + 2
            res_names.append(f"Prot-{parm.residues[res_idx].name.capitalize()}{res_id}")
        elif 1365 <= res_idx <= 1368:
            res_id = res_idx + 2
            res_names.append(f"MGs-{parm.residues[res_idx].name.capitalize()}{res_id}")
        elif 1369 <= res_idx <= 1466:
            res_id = res_idx + 2
            res_names.append(f"RNA-{parm.residues[res_idx].name.capitalize()}{res_id}")
        elif 1467 <= res_idx <= 1515:
            res_id = res_idx + 2
            res_names.append(f"DNA-{parm.residues[res_idx].name.capitalize()}{res_id}")

    res_names.append("Near")
    res_names.append("Far")
    return resid, num_prot_res, res_names


def sum_by_residue(force, resid, num_prot_res, nearforce=None, farforce=None):
    force_by_residue = np.zeros((len(force), 3, 3, num_prot_res + 2))
    for res_idx in range(num_prot_res):
        force_by_residue[:, :, :, res_idx] = force[:, :, :, resid == res_idx].sum(axis=3)

    if nearforce is not None:
        force_by_residue[:, :, :, num_prot_res] = nearforce
    if farforce is not None:
        force_by_residue[:, :, :, num_prot_res + 1] = farforce
    return force_by_residue


def load_qm_forces(prefix):
    forces = []
    for window in range(N_WINDOWS):
        force = (
            -1
            * load_window_array(window, f"{prefix}_forces.npy").swapaxes(1, 2)
            * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        )
        forces.append(force[:, QM_INDEX_EDA])
    return np.array(forces)


def load_electrostatic_residue_forces(prefix, resid, num_prot_res):
    residue_forces = []
    for window in range(N_WINDOWS):
        force = (
            -1
            * load_window_array(window, f"{prefix}_prot_forces.npy")
            * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        )
        nearforce = (
            -1
            * load_window_array(window, f"{prefix}_near_forces.npy")
            * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        )
        farforce = (
            -1
            * load_window_array(window, f"{prefix}_far_forces.npy")
            * (AMBER_HARTREE_TO_KCAL / AMBER_BOHR_TO_A)
        )
        residue_forces.append(sum_by_residue(force, resid, num_prot_res, nearforce, farforce))
    return np.array(residue_forces)


def load_vdw_residue_forces(resid, num_prot_res):
    residue_forces = []
    for window in range(N_WINDOWS):
        # lj_prot_forces.npy keeps the atom dimension required for residue
        # decomposition. lj_prot_forces_sum.npy is already summed over protein
        # atoms and is only appropriate for total protein vdW force workflows.
        force = load_window_array(window, "lj_prot_forces.npy")
        nearforce = load_window_array(window, "lj_near_forces.npy")
        residue_forces.append(sum_by_residue(force, resid, num_prot_res, nearforce))
    return np.array(residue_forces)


def main():
    parm = AmberParm(str(PARM_PATH))
    resid, num_prot_res, res_names = build_residue_metadata(parm)

    save_array("res_names.npy", res_names)
    save_array("qmmm_qm_forces.npy", load_qm_forces("qmmm"))
    save_array("qmmm_res_forces2.npy", load_electrostatic_residue_forces("qmmm", resid, num_prot_res))
    save_array("gas_qm_forces.npy", load_qm_forces("gas"))
    save_array("gas_res_forces2.npy", load_electrostatic_residue_forces("gas", resid, num_prot_res))
    save_array("vdw_res_forces2.npy", load_vdw_residue_forces(resid, num_prot_res))


if __name__ == "__main__":
    main()
