"""
MBAR preprocessing & PMF computation (no plotting).

Key behaviors (kept & improved):
- Auto-detects:
  * val_min from ../00/cv.rst (r2/r3 of the first &rst block)
  * val_max from ../{n_windows-1}/cv.rst
  * force constant (fc) = 2 * rk2 (first &rst block of ../input/cv.rst)
  * nbins = n_windows - 1 (unless overridden)
- Default: ALL REPLICAS (-a/--all-reps is ON by default). Use --no-all-reps to disable.
  When using all replicas, rep is set to '??' for globbing, and output tag becomes 'all'.
- Robust PMF binning: automatically shrinks the PMF range to observed data and/or
  reduces nbins until all bins contain samples (avoids pymbar ParameterError).
- Saves into time-organized subfolders:
  * freefile/{YYYYmmdd-HHMMSS}/freefile_mbar_{step}.all
  * entropies/{YYYYmmdd-HHMMSS}/reweighting_entropy_{step}.all
- Conda env: "mbar"
- Uses your patched pymbar.mbar_pmf (fallback to mbar_pmf2 if needed).

CLI overrides now available: --val-min, --val-max, --fc, --nbins
"""
import os
import re
import argparse
import numpy as np
from glob import glob
from datetime import datetime

import pymbar
# Use your patched module if available; fallback to the uploaded file.
try:
    from pymbar.mbar_pmf import mbar_pmf
except Exception:
    from mbar_pmf2 import mbar_pmf  # fallback for development

# --------------------------- Helpers ---------------------------

def _read_first_rst_block(path):
    """Return the text of the first &rst ... &end block in a cv.rst file."""
    with open(path, "r") as f:
        lines = f.readlines()
    in_block, block = False, []
    for line in lines:
        if "&rst" in line:
            in_block = True
        if in_block:
            block.append(line)
        if in_block and "&end" in line:
            break
    if not block:
        raise ValueError(f"No &rst block found in {path}")
    return "".join(block)

def _parse_value(pat, text, what, path):
    """Regex parse a numeric value from text; fail clearly if missing."""
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find {what} in {path}")
    return float(m.group(1))

def get_val_from_cv_rst(cv_rst_path):
    """Extract r2 (or r3) from first &rst; ensure r2==r3 if both present."""
    block = _read_first_rst_block(cv_rst_path)
    r2 = _parse_value(r"r2\s*=\s*([-+0-9.eE]+)", block, "r2", cv_rst_path)
    r3 = _parse_value(r"r3\s*=\s*([-+0-9.eE]+)", block, "r3", cv_rst_path)
    if abs(r2 - r3) > 1e-12:
        # They should match for a flat-bottom harmonic center; take r2 but warn.
        print(f"[warn] r2 ({r2}) != r3 ({r3}) in {cv_rst_path}; using r2.")
    return r2

def get_fc_from_input_cv(input_cv_path):
    """Extract rk2 from first &rst of ../input/cv.rst and return 2 * rk2."""
    block = _read_first_rst_block(input_cv_path)
    rk2 = _parse_value(r"rk2\s*=\s*([-+0-9.eE]+)", block, "rk2", input_cv_path)
    return 2.0 * rk2  # per instructions

def autodetect_n_windows(base_dir, step, rep_glob):
    """Detect number of windows by scanning ../??/ for cv.rst or data files."""
    cv_dirs = sorted(glob(os.path.join(base_dir, "[0-9][0-9]", "cv.rst")))
    if cv_dirs:
        idxs = [int(os.path.basename(os.path.dirname(p))) for p in cv_dirs]
        return max(idxs) + 1
    data_paths = glob(os.path.join(base_dir, "[0-9][0-9]", f"{step}.{rep_glob}_equilibration.cv"))
    if data_paths:
        idxs = [int(os.path.basename(os.path.dirname(p))) for p in data_paths]
        return max(idxs) + 1
    raise RuntimeError("Could not autodetect n_windows; provide --n-windows.")

def _time_bucket_dir(root_dir):
    """Create and return a timestamped subdirectory under root_dir."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(root_dir, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _determine_pmf_binning(val_kn, seed_min, seed_max, seed_nbins, min_nbins=5):
    """
    Choose (val_min, val_max, nbins) so every bin has at least one sample.
    - Start from (seed_min, seed_max, seed_nbins)
    - Clamp the range to observed data to avoid empty edge bins
    - If any interior bins are empty, decrement nbins until all occupied (>= min_nbins)
    Returns: (val_min, val_max, nbins)
    """
    data = np.concatenate(val_kn)
    obs_min, obs_max = float(np.min(data)), float(np.max(data))

    # Clamp to observed range (avoid bins entirely outside data support)
    val_min = max(seed_min, obs_min)
    val_max = min(seed_max, obs_max)
    if not np.isfinite(val_min) or not np.isfinite(val_max) or val_min >= val_max:
        val_min, val_max = obs_min, obs_max

    nbins = int(seed_nbins)
    if nbins < 1:
        nbins = max(1, int((len(data) ** 0.5)))  # fallback: sqrt(N) rule

    def _has_empty(n):
        counts, _ = np.histogram(data, bins=n, range=(val_min, val_max))
        return np.any(counts == 0)

    # Reduce nbins if necessary to eliminate empties
    while nbins > min_nbins and _has_empty(nbins):
        nbins -= 1

    # Last resort: if still empty bins at min_nbins, shrink range slightly within observed support
    tries = 0
    while _has_empty(nbins) and tries < 5:
        # Trim 0.5% tails each side
        qlo, qhi = np.quantile(data, [0.005, 0.995])
        val_min = max(val_min, float(qlo))
        val_max = min(val_max, float(qhi))
        tries += 1

    if _has_empty(nbins):
        # If we still have empties, collapse to  min_nbins=3 to proceed
        nbins = max(3, min_nbins)
        # Final clamp to min/max data
        val_min, val_max = float(np.min(data)), float(np.max(data))

    print(f"[info] PMF binning: val_min={val_min:.6f}, val_max={val_max:.6f}, nbins={nbins}")
    return val_min, val_max, nbins

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute PMF with MBAR (no plotting)." )
    parser.add_argument("-s", "--step", default="step6", help="Simulation step tag (default: step5)")
    parser.add_argument("-r", "--rep", default="00", help="Replica tag if not using all-reps (default: 00)")
    parser.add_argument("-a", "--all-reps", dest="all_reps", action="store_true", default=True,
                        help="Use all replicas (default: ON)." )
    parser.add_argument("--no-all-reps", dest="all_reps", action="store_false",
                        help="Disable all-reps; use only --rep.")
    parser.add_argument("-nw", "--n-windows", type=int, default=None, help="Number of windows; autodetect if omitted")
    parser.add_argument("--base-dir", default="..", help="Base directory containing window subdirs (default: ..)" )
    parser.add_argument("--input-cv", default="../input/cv.rst", help="Path to ../input/cv.rst")
    parser.add_argument("-t", "--temperature", type=float, default=300.0, help="Temperature in K (default: 300.0)")
    parser.add_argument("--freefile-dir", default="freefile", help="Output dir root for freefile (default: freefile)" )
    parser.add_argument("--entropies-dir", default="entropies", help="Output dir root for entropies (default: entropies)" )
    # New overrides
    parser.add_argument("--val-min", dest="val_min_override", type=float, default=None, help="Override PMF val_min" )
    parser.add_argument("--val-max", dest="val_max_override", type=float, default=None, help="Override PMF val_max" )
    parser.add_argument("--fc", dest="fc_override", type=float, default=None, help="Override force constant (kcal/mol/Å^2)" )
    parser.add_argument("-nb", "--nbins", dest="nbins_override", type=int, default=None, help="Override number of PMF bins" )
    args = parser.parse_args()

    # Preserve original names
    step = args.step
    rep = args.rep
    rep_glob = "??" if args.all_reps else rep
    out_tag = "all" if args.all_reps else rep

    n_windows = args.n_windows if args.n_windows is not None else autodetect_n_windows(args.base_dir, step, rep_glob)

    # Auto-derived quantities
    val_min_auto = get_val_from_cv_rst(os.path.join(args.base_dir, "00", "cv.rst"))
    val_max_auto = get_val_from_cv_rst(os.path.join(args.base_dir, f"{n_windows-1:02d}", "cv.rst"))
    fc_auto = get_fc_from_input_cv(args.input_cv)     # force constant = 2 * rk2
    nbins_auto = n_windows - 1                        # per instruction

    # Apply overrides if provided
    val_min = args.val_min_override if args.val_min_override is not None else val_min_auto
    val_max = args.val_max_override if args.val_max_override is not None else val_max_auto
    fc = args.fc_override if args.fc_override is not None else fc_auto
    nbins = args.nbins_override if args.nbins_override is not None else nbins_auto
    temperature = args.temperature

    # Derived arrays (unchanged variable names)
    val0_k = np.linspace(val_min_auto, val_max_auto, n_windows)  # centers still based on window design
    K_k = np.ones(n_windows) * fc

    # Gather CV samples per window (unchanged input pattern)
    val_kn = []
    for i in range(n_windows):
        fnames = sorted(glob(os.path.join(args.base_dir, f"{i:02d}", f"{step}.{rep_glob}_equilibration.cv")))
        if not fnames:
            raise FileNotFoundError(
                f"No CV file found for window {i:02d}: "
                f"{os.path.join(args.base_dir, f'{i:02d}', f'{step}.{rep_glob}_equilibration.cv')}"
            )
        arrays = [np.loadtxt(f, usecols=1) for f in fnames]
        val_kn.append(np.concatenate(arrays))

    # Echo correlation info (kept from original)
    for i in range(n_windows):
        ess_idx = pymbar.timeseries.subsampleCorrelatedData(val_kn[i], conservative=True)
        print(f"Window {i:02d}: {len(ess_idx)} effective samples")

    # Robust PMF bin selection to avoid empty bins
    pmf_min, pmf_max, pmf_nbins = _determine_pmf_binning(val_kn, val_min, val_max, nbins)

    # MBAR with your custom class
    mbar = mbar_pmf(val_kn, val0_k, K_k, temperature)

    # PMF & uncertainties
    bin_centers, f_i, df_i, reweighting_entropy = mbar.get_pmf(pmf_min, pmf_max, pmf_nbins)
    # Stabilize PMF reference similarly
    pmf_ref = f_i[:20].argmin()
    bin_centers, f_i, df_i, reweighting_entropy = mbar.get_pmf(
        pmf_min, pmf_max, pmf_nbins, uncertainties='from-specified', pmf_reference=pmf_ref
    )

    # Ensure time-organized output directories exist
    freefile_run_dir = _time_bucket_dir(args.freefile_dir)
    ent_run_dir = _time_bucket_dir(args.entropies_dir)

    # Write outputs (filenames use '.all' when aggregating all replicas)
    freefile_path = os.path.join(freefile_run_dir, f"freefile_mbar_{step}.{out_tag}")
    np.savetxt(freefile_path, np.column_stack((bin_centers, f_i, df_i)))
    print(f"[ok] Wrote PMF to {freefile_path}")

    ent_path = os.path.join(ent_run_dir, f"reweighting_entropy_{step}.{out_tag}")
    np.savetxt(ent_path, np.column_stack((bin_centers, reweighting_entropy)))
    print(f"[ok] Wrote reweighting entropies to {ent_path}")

if __name__ == "__main__":
    main()
