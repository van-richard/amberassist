# md_tools/topology.py

import os
from glob import glob


def find_parm(base_dir=None):
	"""
	Automatically detect parm7 file.

	Priority:
		1) step3_pbcsetup_*.parm7
		2) step3_pbcsetup.parm7

	Parameters
	----------
	base_dir : str, optional
		Directory to search. Defaults to current working directory.

	Returns
	-------
	str
		Path to detected parm7 file.

	Raises
	------
	RuntimeError
		If no suitable parm7 file is found.
	"""

	if base_dir is None:
		base_dir = os.getcwd()

	# Look for versioned parm first
	versioned = sorted(
			glob(os.path.join(base_dir, "step3_pbcsetup_*.parm7"))
			)

	if versioned:
		if len(versioned) > 1:
			raise RuntimeError(
					f"Multiple versioned parm files found: {versioned}"
					)
		return versioned[0]

	# Fallback to default parm
	default = os.path.join(base_dir, "step3_pbcsetup.parm7")

	if os.path.exists(default):
		return default

	# Nothing found
	raise RuntimeError(
			"No parm7 file found. Please provide --parm explicitly."
			)

