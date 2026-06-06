# coordination.py

import numpy as np

try:
    import pytraj as pt
except ImportError:
    raise ImportError("pytraj required. Load AmberTools module.")

from .qm import get_qm_residues


def get_qm_residue_info(traj, directory='../00', pattern='step5.00*mdin'):

    qm_residues = get_qm_residues(directory, pattern)

    top = traj.top
    qm_resnames = []

    for res in qm_residues:
        selection = top.select(f':{res}')

        if len(selection) == 0:
            qm_resnames.append("UNKNOWN")
        else:
            atom = top.atom(selection[0])
            qm_resnames.append(atom.resname)

    return qm_residues, qm_resnames


def detect_metal_atom_index(traj, qm_residues, qm_resnames):

    top = traj.top
    metal_names = {'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'CA', 'NA+'}

    for res, name in zip(qm_residues, qm_resnames):

        if name.upper() in metal_names:

            selection = top.select(f':{res}')
            if len(selection) == 0:
                raise RuntimeError(f"No atoms found in metal residue {res}")

            return selection[0] + 1, res

    raise RuntimeError("No metal residue detected in QM region.")


def generate_mecs(traj, metal_index, qm_residues, cutoff=3.0):

    top = traj.top

    metal_atom = top.atom(metal_index - 1)
    metal_residue = metal_atom.resid + 1

    mecs = []
    meclabels = []

    for res in qm_residues:

        if res == metal_residue:
            continue

        atom_indices = top.select(f':{res}&!@H=')
        if len(atom_indices) == 0:
            continue

        masks = [f'@{metal_index} @{idx+1}' for idx in atom_indices]

        dist = pt.distance(traj, mask=masks)
        dist = np.asarray(dist)

        if dist.ndim == 1:
            avg_dist = np.array([np.mean(dist)])
        else:
            avg_dist = np.mean(dist, axis=1)

        min_idx = np.argmin(avg_dist)
        min_dist = avg_dist[min_idx]

        if min_dist <= cutoff:

            selected_atom = atom_indices[min_idx]
            mecs.append(selected_atom + 1)

            atom = top.atom(selected_atom)
            meclabels.append(f"{atom.resname}({atom.name})")

    return mecs, meclabels

def build_mecs_masks(metal_residue, mecs):
    return [f':{metal_residue} @{idx}' for idx in mecs]

