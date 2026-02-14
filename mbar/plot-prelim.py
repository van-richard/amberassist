"""
Standalone plotting for the MBAR results produced by mbar0.py.

Reads:
  - freefile/*/freefile_mbar_{step}.all (by default; or {step}.{rep} if --no-all-reps)
  - ../??/{step}.{rep_glob}_equilibration.cv for hist/kde overlays
  - ../00/cv.rst and ../{n_windows-1}/cv.rst to rebuild val0_k for tick/lines

Saves:
  - img/{YYYYmmdd-HHMMSS}/prelim-{step}.all.png  (or ...{step}.{rep}.png)
Default: -a/--all-reps is ON; use --no-all-reps with --rep to target a single replica.
"""
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from glob import glob
from datetime import datetime

# --------------------------- Helpers (duplicated to keep script standalone) ---------------------------

def _read_first_rst_block(path):
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
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find {what} in {path}")
    return float(m.group(1))

def get_val_from_cv_rst(cv_rst_path):
    block = _read_first_rst_block(cv_rst_path)
    r2 = _parse_value(r"r2\s*=\s*([-+0-9.eE]+)", block, "r2", cv_rst_path)
    r3 = _parse_value(r"r3\s*=\s*([-+0-9.eE]+)", block, "r3", cv_rst_path)
    if abs(r2 - r3) > 1e-12:
        print(f"[warn] r2 ({r2}) != r3 ({r3}) in {cv_rst_path}; using r2.")
    return r2

def autodetect_n_windows(base_dir, step, rep_glob):
    cv_dirs = sorted(glob(os.path.join(base_dir, "[0-9][0-9]", "cv.rst")))
    if cv_dirs:
        idxs = [int(os.path.basename(os.path.dirname(p))) for p in cv_dirs]
        return max(idxs) + 1
    data_paths = glob(os.path.join(base_dir, "[0-9][0-9]", f"{step}.{rep_glob}_equilibration.cv"))
    if data_paths:
        idxs = [int(os.path.basename(os.path.dirname(p))) for p in data_paths]
        return max(idxs) + 1
    raise RuntimeError("Could not autodetect n_windows; provide --n-windows.")

def _latest_subdir(root):
    """Return the most recently modified direct subdirectory of root, or None."""
    if not os.path.isdir(root):
        return None
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return subdirs[0]

def _resolve_freefile_path(root, filename):
    """Find filename in root or latest timestamped subdir under root."""
    direct = os.path.join(root, filename)
    if os.path.isfile(direct):
        return direct
    latest = _latest_subdir(root)
    if latest is not None:
        candidate = os.path.join(latest, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {filename} in {root} or its latest subdir.")

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot MBAR PMF and histograms.")
    parser.add_argument("--step", default="step5", help="Simulation step tag (default: step5)")
    parser.add_argument("--rep", default="00", help="Replica tag (used when --no-all-reps)")
    parser.add_argument("-a", "--all-reps", dest="all_reps", action="store_true", default=True,
                        help="Use all replicas (default: ON).")
    parser.add_argument("--no-all-reps", dest="all_reps", action="store_false",
                        help="Disable all-reps; use only --rep.")
    parser.add_argument("--n-windows", type=int, default=None, help="Number of windows; autodetect if omitted")
    parser.add_argument("--base-dir", default="..", help="Base directory containing window subdirs (default: ..)")
    parser.add_argument("--freefile-dir", default="freefiles", help="Dir containing freefile outputs (default: freefile)")
    parser.add_argument("--img-dir", default="img", help="Directory root for output figures (default: img)")
    args = parser.parse_args()

    step = args.step
    rep = args.rep
    rep_glob = "??" if args.all_reps else rep
    out_tag = "all" if args.all_reps else rep

    n_windows = args.n_windows if args.n_windows is not None else autodetect_n_windows(args.base_dir, step, rep_glob)

    # Rebuild val0_k from cv.rst (consistent with mbar0.py)
    val_min = get_val_from_cv_rst(os.path.join(args.base_dir, "00", "cv.rst"))
    val_max = get_val_from_cv_rst(os.path.join(args.base_dir, f"{n_windows-1:02d}", "cv.rst"))
    val0_k = np.linspace(val_min, val_max, n_windows)

    # Load PMF result (resolve latest timestamped bucket automatically)
    freefile_name = f"freefile_mbar_{step}.{out_tag}"
    freefile_path = _resolve_freefile_path(args.freefile_dir, freefile_name)
    initial = np.loadtxt(freefile_path)
    xdata = initial[:, 0]
    ydata = initial[:, 1] - initial[:10, 1].min()
    edata = initial[:, 2]

    dgd = round(initial[:, 1].max() - initial[:10, 1].min(), 1)            # ΔG‡
    err = round(initial[initial[:, 1].argmax()][2], 1)                     # MBAR error

    # Load samples for hist/kde
    val_kn = []
    for i in range(n_windows):
        fnames = sorted(glob(os.path.join(args.base_dir, f"{i:02d}", f"{step}.{rep_glob}_equilibration.cv")))
        arrays = [np.loadtxt(f, usecols=1) for f in fnames]
        val_kn.append(np.concatenate(arrays))

    # Time-organized image directory
    ts_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_img_dir = os.path.join(args.img_dir, ts_dir)
    os.makedirs(out_img_dir, exist_ok=True)

    # ---------------- Plot ----------------
    fig = plt.figure(figsize=(10, 6))
    axs = fig.subplot_mosaic(
        """
        A
        B
        """, height_ratios=[1, 2], sharex=True
    )

    c = sb.color_palette('deep', n_windows)
    opa = 0.4
    _xfrac, _yfrac = 0.05, 0.9

    for i in range(n_windows):
        sb.kdeplot(val_kn[i], fill=True, alpha=opa, ax=axs['A'], color=c[i])
        axs['A'].axvline(x=val0_k[i], linestyle='--', alpha=opa, color=c[i])
        axs['A'].yaxis.get_major_ticks()[0].label1.set_visible(False)
        axs['A'].grid(linestyle='-', alpha=opa-0.2)

        axs['B'].errorbar(xdata, ydata, yerr=edata, linewidth=1, c='black', alpha=opa+0.1)
        axs['B'].scatter(xdata[i-1], ydata[i-1], color=c[i])
        axs['B'].axvline(x=val0_k[i], linestyle='--', alpha=opa, color=c[i])
        axs['B'].grid(linestyle='-', alpha=opa-0.2)

    axs['B'].annotate(f"freefile: {freefile_path}",
                      xy=(_xfrac, _yfrac), xycoords='axes fraction',
                      bbox=dict(fc="w", alpha=opa))
    axs['B'].annotate(f"$\\Delta G^\\ddag$ = {dgd} $\\pm$ {err}",
                      xy=(_xfrac, _yfrac-0.15), xycoords='axes fraction',
                      bbox=dict(fc='w', alpha=opa+0.2))

    axs['B'].set_xlabel("r1 - r2 (Å)")
    axs['B'].set_ylabel("Potential of Mean Force (kcal/mol)")

    plt.margins(x=0.00, y=0.1)
    plt.xticks(ticks=val0_k, rotation=55, ha='right')

    sb.despine(left=True, bottom=False, right=True, ax=axs['A'])
    fig.subplots_adjust(wspace=0, hspace=0)

    out_png = os.path.join(out_img_dir, f"prelim-{step}.{out_tag}.png")
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"[ok] Saved figure to {out_png}")

if __name__ == "__main__":
    main()
