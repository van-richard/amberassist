# python/qmmm_tools/distances.py

import os
import numpy as np

try:
    import pytraj as pt
except ImportError:
    raise ImportError(
        "pytraj is required for qmmm_tools.distances. "
        "Load AmberTools module or install AmberTools."
    )

from .windows import find_windows, find_parm


def save_distances(avgs, stds, step_name, masktype, base_dir=None):
    """
    Save distance statistics to pickle file.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    os.makedirs(os.path.join(base_dir, 'pickle'), exist_ok=True)

    workdir = os.path.basename(os.path.dirname(base_dir))
    dname = f'{workdir}-{step_name}-{masktype}'

    results = {'avgs': avgs, 'stds': stds}

    outfile = os.path.join(base_dir, 'pickle', f'distances-{dname}.pkl')
    pt.to_pickle(results, outfile)


def calc_distances(
    step,
    ambermask,
    masktype,
    base_dir=None,
    save=False,
    verbose=False
):
    """
    Calculate distance averages and std per window.

    Parameters
    ----------
    step : str
        Step name (e.g., 'step6')
    ambermask : str or list
        Amber mask(s) for pt.distance
    masktype : str
        Label for output naming
    base_dir : str, optional
        Working directory
    save : bool
    verbose : bool

    Returns
    -------
    tuple
        (avgs, stds)
    """

    if base_dir is None:
        base_dir = os.getcwd()

    n_windows, windows = find_windows(base_dir)
    parm = find_parm(base_dir)

    avgs = []
    stds = []

    for iw, w in enumerate(windows):

        nc = os.path.join(w, f'{step}.*_equilibration.nc')
        traj = pt.iterload(nc, parm)

        data = pt.distance(traj, mask=ambermask)

        data = np.asarray(data)

        if data.ndim == 1:
            data = data[:, np.newaxis]

        if verbose:
            print(f'Processing window {iw+1}/{n_windows}: {w}')
            print('  data shape:', data.shape)

        avgs.append(np.mean(data, axis=1))
        stds.append(np.std(data, axis=1))

        del traj
        del data

    avgs = np.array(avgs)
    stds = np.array(stds)

    if save:
        save_distances(avgs, stds, step, masktype, base_dir)

    return avgs, stds


def read_distances(xdata, fname, verbose=False):
    """
    Load saved distance pickle and validate shape.
    """
    results = pt.read_pickle(fname)

    avgs = results['avgs']
    stds = results['stds']

    if verbose:
        print('avgs shape:', avgs.shape)
        print('stds shape:', stds.shape)

    assert avgs.shape == stds.shape
    assert avgs.shape[0] == len(xdata)

    return avgs, stds

