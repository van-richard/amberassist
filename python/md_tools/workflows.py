# md_tools/workflows.py


def print_box_info(traj_files, parm_file):
    try:
        import pytraj as pt
    except ImportError:
        raise ImportError(
            "pytraj required. Load AmberTools module."
        )
    traj = pt.iterload(traj_files, parm_file)
    return print(traj[-1].box)
    

def count_frames_from_files(traj_files, parm_file):

    try:
        import pytraj as pt
    except ImportError:
        raise ImportError(
            "pytraj required. Load AmberTools module."
        )

    from .trajectory import count_frames

    results = {}

    for fname in traj_files:

        traj = pt.iterload(fname, parm_file)
        n = count_frames(traj)

        results[fname] = n

        del traj

    return results


def summarize_trajectory(traj):
    """
    Return summary information about trajectory.
    """
    from .trajectory import count_frames

    summary = {}

    summary["n_frames"] = count_frames(traj)

    try:
        summary["n_atoms"] = traj.top.n_atoms
    except AttributeError:
        summary["n_atoms"] = None

    return summary


def load_and_summarize(traj_file, parm_file):
    """
    Load trajectory using pytraj and summarize it.
    """
    try:
        import pytraj as pt
    except ImportError:
        raise ImportError(
            "pytraj required to load trajectory. "
            "Load AmberTools module."
        )

    traj = pt.iterload(traj_file, parm_file)

    return summarize_trajectory(traj)

