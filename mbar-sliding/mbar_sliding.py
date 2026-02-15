import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
from sklearn.utils import resample

import pymbar
from pymbar.mbar_pmf import mbar_pmf

os.makedirs('img', exist_ok=True)
os.makedirs('freefiles', exist_ok=True)
os.makedirs('entropies', exist_ok=True)


steprep=sys.argv[1]
step = steprep.split('.')[0]
rep = steprep.split('.')[1]

n_windows = int(sys.argv[2])
val_min = float(sys.argv[3]) # restraint at window 00
val_max = float(sys.argv[4]) # restraint at window 41

val_step = 0.1
fc = 300.0      
nbins = n_windows - 1

val0_k = np.arange(val_min, val_max+val_step, val_step)
K_k = np.ones(n_windows) * fc

val_kn = np.load('cvs.npy')

# ---------------- Sliding-window settings ----------------
total_ps = float(sys.argv[5])     # total sampling length per window
win_ps   = float(sys.argv[6])     # window length
shift_ps = float(sys.argv[7])     # shift between windows
start_ps = float(sys.argv[8])     # optional: e.g. 5.0 to skip first 5 ps
stop_ps  = total_ps # optional: can shorten, e.g. 45.0

K = n_windows
N = val_kn.shape[1]  # samples per window
ps_per_sample = total_ps / float(N)

win_len   = int(round(win_ps   / ps_per_sample))
shift_len = int(round(shift_ps / ps_per_sample))
start_i   = int(round(start_ps / ps_per_sample))
stop_i    = int(round(stop_ps  / ps_per_sample))

if win_len < 2:
    raise ValueError("win_ps too small relative to sampling interval.")
if shift_len < 1:
    shift_len = 1
if stop_i > N:
    stop_i = N
if start_i < 0:
    start_i = 0
if start_i + win_len > stop_i:
    raise ValueError("No valid sliding windows: check start_ps/stop_ps/win_ps.")

starts = np.arange(start_i, stop_i - win_len + 1, shift_len, dtype=int)

print(f"Samples/window N={N}, ps/sample={ps_per_sample:.6f}")
print(f"Sliding: win_len={win_len} samples ({win_ps} ps), shift_len={shift_len} samples ({shift_ps} ps)")
print(f"Number of segments: {len(starts)}")


# ---------------- Choose ONE consistent PMF reference bin (recommended) ----------------
# Compute full PMF once to choose a stable pmf_reference bin (like you currently do with first 20 bins).
mbar_full = mbar_pmf(val_kn, val0_k, K_k, fc)
bin_centers_full, f_full, df_full, S_full = mbar_full.get_pmf(val_min, val_max, nbins)
pmf_ref_bin = f_full[:20].argmin()  # same rule you already use


# ---------------- Sliding-window PMFs ----------------
for s in starts:
    e = s + win_len

    # segment CVs: shape (K, win_len)
    val_kn_seg = val_kn[:, s:e]

    # run MBAR PMF for this segment
    mbar_seg = mbar_pmf(val_kn_seg, val0_k, K_k, fc)

    # same binning range/nbins; anchor to the SAME reference bin for all segments
    bin_centers, f_i, df_i, reweighting_entropy = mbar_seg.get_pmf(
            val_min, val_max, nbins,
            uncertainties='from-specified',
            pmf_reference=pmf_ref_bin
            )

    t0 = s * ps_per_sample
    t1 = e * ps_per_sample

    # safe-ish tag for filenames (avoid '?' in case it annoys your shell later)
    rep_tag = rep.replace('?', 'q')

    out_ff = f"freefiles/freefile_mbar_{step}.{rep_tag}_t{t0:05.1f}-{t1:05.1f}ps"
    out_re = f"entropies/reweighting_entropy_{step}.{rep_tag}_t{t0:05.1f}-{t1:05.1f}ps"
    np.savetxt(out_ff, np.column_stack((bin_centers, f_i, df_i)))
    np.savetxt(out_re, reweighting_entropy)

    print(f"Wrote {out_ff} and {out_re}")

