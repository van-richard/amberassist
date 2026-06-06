# python/qmmm_tools/windows.py

import os
import sys
from glob import glob
import numpy as np


def find_basename(base_dir=None):
    """
    Return parent directory name of working directory.

    Parameters
    ----------
    base_dir : str, optional
        Base directory. Defaults to current working directory.

    Returns
    -------
    str
    """
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.basename(os.path.dirname(base_dir))


def find_windows(base_dir=None, pattern='[0-9][0-9]'):
    """
    Find umbrella sampling windows.

    Parameters
    ----------
    base_dir : str, optional
        Base directory containing windows.
    pattern : str
        Glob pattern for window directories.

    Returns
    -------
    tuple
        (n_windows, windows_list)
    """
    if base_dir is None:
        base_dir = os.getcwd()

    search_path = os.path.join(base_dir, '..', pattern)
    windows = sorted(
        [d for d in glob(search_path) if os.path.isdir(d)]
    )

    return len(windows), windows


def find_parm(base_dir=None):
    """
    Find single parm7 file in ../input directory.

    Parameters
    ----------
    base_dir : str, optional

    Returns
    -------
    str
        Path to parm7 file.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    search_path = os.path.join(base_dir, '..', 'input', '*.parm7')
    parm_files = glob(search_path)

    if len(parm_files) != 1:
        raise RuntimeError(
            f"Expected exactly one parm7 file in ../input/, found {len(parm_files)}"
        )

    return parm_files[0]


def find_cv_min(cv_fname='cv.rst', windows=None):
    """
    Extract r2 value from first window cv.rst file.

    Returns
    -------
    float
    """
    if windows is None:
        _, windows = find_windows()

    if len(windows) == 0:
        raise RuntimeError("No windows found.")

    cv_file = os.path.join(windows[0], cv_fname)

    cv_min = None
    with open(cv_file, 'r') as f:
        for line in f:
            if 'r1=' in line:
                parts = line.strip().split(',')
                for part in parts:
                    if 'r2=' in part:
                        cv_min = float(part.split('=')[1])
                        break
                break

    if cv_min is None:
        raise RuntimeError(f"Could not find r2= in {cv_file}")

    return cv_min


def get_xdata(n_windows=None, cv_min=None):
    """
    Generate x-axis values for umbrella sampling.

    Returns
    -------
    np.ndarray
    """
    if n_windows is None:
        n_windows, windows = find_windows()
    else:
        windows = None

    if cv_min is None:
        cv_min = find_cv_min(windows=windows)

    return cv_min + 0.1 * np.arange(n_windows)

