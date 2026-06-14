#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np


methods = [
    # "b3lyp_6-31gd",
    # "b3lyp_6-31+gd",
    # "blyp_6-31gd",
    # "blyp_6-31+gd",
    # "wb97xd_6-31gd",
    "wb97xd_6-31+gd",
]

n_windows = 42
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
energy_root = script_dir / "qmmm_energies"
mbar_tp_energy_dir = root_dir / "mbar" / "tp_energy"


def combine_window_energy(method, window):
    method_dir = energy_root / method / f"{window:02d}"
    fnames = sorted(method_dir.glob("*/qmmm_%s_energy.npy" % method))
    if not fnames:
        raise FileNotFoundError(f"No frame energy files found in {method_dir}")

    energy = []
    for fname in fnames:
        energy.append(np.load(fname))

    energies = np.concatenate(energy)
    print(method, f"{window:02d}", energies.shape)

    output = method_dir / ("qmmm_%s_energy_all.npy" % method)
    np.save(output, energies)
    return output


def combine_method_energy(method):
    per_window = []
    for window in range(n_windows):
        per_window.append(combine_window_energy(method, window))

    arr = [np.load(fname)[::] for fname in per_window]
    energies = np.column_stack(arr)

    mbar_tp_energy_dir.mkdir(parents=True, exist_ok=True)
    output = mbar_tp_energy_dir / ("qmmm_%s_energy.npy" % method)
    np.save(output, np.swapaxes(energies, 0, 1))
    print(output)


for method in methods:
    combine_method_energy(method)
