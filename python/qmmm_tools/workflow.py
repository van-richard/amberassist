# qmmm_tools/workflows.py

from .qm import get_qm_residues
from .coordination import (
		get_qm_residue_info,
		detect_metal_atom_index,
		generate_mecs
		)
from .rc import (
		generate_rcs_from_cv,
		build_extra_rcs,
		build_rclabels
		)


def generate_rcs_and_mecs(
		traj,
		cv_file,
		qm_directory='../00',
		cutoff=3.0,
		extra_pairs=None
		):
	"""
	High-level workflow wrapper.

	Returns dict with:
		qm_residues
		qm_resnames
		metal_index
		metal_residue
		mecs
		meclabels
		rcs
		rclabels
	"""

	qm_residues, qm_resnames = get_qm_residue_info(
			traj,
			directory=qm_directory
			)

	metal_index, metal_residue = detect_metal_atom_index(
			traj,
			qm_residues,
			qm_resnames
			)

	mecs, meclabels = generate_mecs(
			traj,
			metal_index,
			qm_residues,
			cutoff
			)

	rcs = generate_rcs_from_cv(cv_file)

	if extra_pairs:
		rcs.extend(build_extra_rcs(traj.top, extra_pairs))

	rclabels = build_rclabels(traj.top, rcs)

	return {
			"qm_residues": qm_residues,
			"qm_resnames": qm_resnames,
			"metal_index": metal_index,
			"metal_residue": metal_residue,
			"mecs": mecs,
			"meclabels": meclabels,
			"rcs": rcs,
			"rclabels": rclabels
			}

