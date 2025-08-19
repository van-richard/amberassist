#!/usr/bin/env python3
"""
mbar_cli.py – MBAR analysis of umbrella sampling data
Uses custom mbar_pmf2.mbar_pmf (with reweighting entropy support).
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
from pymbar import timeseries

# Import the modified class from local mbar_pmf2.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbar_pmf2 import mbar_pmf

import argparse
parser = argparse.ArgumentParser(description="MBAR PMF analysis")
parser.add_argument("-s", "--step", default="step5", help="Simulation step prefix")
parser.add_argument("-r", "--rep", default="00", help="Replica ID")
parser.add_argument("-n", "--n-windows", type=int, default=42, help="Number of umbrella windows")
parser.add_argument("--val-min", type=float, default=-1.10, help="Minimum CV value")
parser.add_argument("--val-max", type=float, default=3.00, help="Maximum CV value")
parser.add_argument("-T", "--temperature", type=float, default=300.0, help="Temperature (K)")
parser.add_argument("-k", "--k-spring", type=float, default=300.0, help="Umbrella spring constant (kcal/mol/Å^2)")
parser.add_argument("-b", "--nbins", type=int, default=None, help="Number of histogram bins (default: n_windows-1)")
parser.add_argument("-o", "--out-prefix", default=None, help="Output file prefix")
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

step, rep = args.step, args.rep
n_windows = args.n_windows
val_min, val_max = args.val_min, args.val_max
temperature_K = args.temperature
k_spring = args.k_spring
nbins = args.nbins or (n_windows - 1)
out_prefix = args.out_prefix or f"mbar_{step}.{rep}"

# Setup umbrella centers and force constants
val0_k = np.linspace(val_min, val_max, n_windows)
K_k = np.ones(n_windows) * k_spring

# Read CV files
val_kn = []
for i in range(n_windows):
    fnames = sorted(glob(f'../{i:02d}/{step}.{rep}_equilibration.cv'))
    if not fnames:
        raise FileNotFoundError(f"No CV files for window {i:02d}")
    arrays = [np.loadtxt(f, usecols=1) for f in fnames]
    vals = np.concatenate(arrays)

    # decorrelate
    g = timeseries.statisticalInefficiency(vals)
    idx = timeseries.subsampleCorrelatedData(vals, g=g)
    vals = vals[idx]
    logging.info(f"Window {i:02d}: kept {len(vals)} (g={g:.1f})")
    val_kn.append(vals)

# Run MBAR
mbar = mbar_pmf(val_kn, val0_k, K_k, temperature_K)
bin_centers, f_i, df_i, S_i = mbar.get_pmf(val_min, val_max, nbins)

# Choose PMF reference as global min
pmf_ref = np.argmin(f_i)
bin_centers, f_i, df_i, S_i = mbar.get_pmf(
    val_min, val_max, nbins,
    uncertainties="from-specified",
    pmf_reference=pmf_ref
)

# Save free energies
np.savetxt(f"freefile_{out_prefix}.dat", np.column_stack((bin_centers, f_i, df_i)))

# Plot
os.makedirs("img", exist_ok=True)
c = sb.color_palette("deep", n_windows)
fig = plt.figure(figsize=(9,6), dpi=160)
axs = fig.subplot_mosaic("A\nB", height_ratios=[1,2], sharex=True)

# Window histograms
opa = 0.4
for i in range(n_windows):
    sb.histplot(val_kn[i], bins=40, alpha=opa, ax=axs['A'], color=c[i])
    axs['A'].axvline(val0_k[i], ls="--", alpha=opa, color=c[i])

# PMF with errors
axs['B'].errorbar(bin_centers, f_i - f_i.min(), yerr=df_i, lw=1, c="k", capsize=2)
axs['B'].set_xlabel("CV (Å)")
axs['B'].set_ylabel("PMF (kcal/mol)")
axs['A'].set_ylabel("Counts")

# Annotate ΔG‡ if barrier exists
left_mask = bin_centers < (val_min + 0.25*(val_max-val_min))
center_mask = (bin_centers > (val_min+0.3*(val_max-val_min))) & (bin_centers < (val_min+0.7*(val_max-val_min)))
if np.any(left_mask) and np.any(center_mask):
    dG_dagger = (np.max(f_i[center_mask]) - np.min(f_i[left_mask]))
    err = df_i[center_mask][np.argmax(f_i[center_mask])]
    axs['B'].annotate(rf"$\Delta G^\ddag = {dG_dagger:.2f} \pm {err:.2f}$",
                      xy=(0.02,0.9), xycoords="axes fraction",
                      bbox=dict(fc="w", alpha=0.5))

plt.tight_layout()
plt.savefig(f"img/{out_prefix}.png")
plt.savefig(f"img/{out_prefix}.pdf")
plt.close()

# Save run metadata
meta = dict(vars(args))
meta.update(dict(pmf_reference=int(pmf_ref)))
with open(f"{out_prefix}.json", "w") as f:
    json.dump(meta, f, indent=2)

logging.info(f"Analysis complete. Results saved to {out_prefix}.*")

