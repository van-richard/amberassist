# md_tools/trajectory.py


def count_frames(traj):
	"""
	Return number of frames in a trajectory-like object.
	"""
	try:
		return traj.n_frames
	except AttributeError:
		return len(traj)


