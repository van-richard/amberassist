#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np

def fidx(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1

def load_one(fp: Path):
    z = np.load(fp, allow_pickle=True)
    rec = {k: z[k] for k in z.files}
    # meta is json string
    rec["meta"] = json.loads(str(rec["meta"]))
    # optional forces
    if "forces" in rec:
        rec["forces"] = rec["forces"].astype(np.float32)
    return rec

def combine_dir(frames_dir: Path, outdir: Path):
    fps = sorted(frames_dir.glob("wtp_frame_*.npz"), key=lambda p: fidx(p.name))
    if not fps:
        raise SystemExit(f"No frames in {frames_dir}")
    idx, E_QM, E_ref, dE = [], [], [], []
    has_forces = False
    forces_list = []
    names = []
    for fp in fps:
        r = load_one(fp)
        i = fidx(fp.name)
        idx.append(i)
        E_QM.append(float(r["E_QM"]))
        E_ref.append(float(r["E_ref"]))
        dE.append(float(r["dE"]))
        names.append(r["meta"].get("inp",""))
        if "forces" in r:
            has_forces = True
            forces_list.append(r["forces"])
    outdir.mkdir(parents=True, exist_ok=True)
    npz = {
            "frame_index": np.array(idx, np.int32),
            "E_QM": np.array(E_QM, np.float64),
            "E_ref": np.array(E_ref, np.float64),
            "dE":   np.array(dE, np.float64),
            }
    if has_forces:
        # Ragged forces would be awkward; assume consistent shape; save stacked
        npz["forces"] = np.stack(forces_list, axis=0).astype(np.float32)
    np.savez_compressed(outdir / "wtp_window_combined.npz", **npz)

    with open(outdir / "wtp_window_index.csv", "w") as f:
        f.write("frame,inp\n")
        for i, n in zip(idx, names):
            f.write(f"{i},{n}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="Dir with per-frame npz")
    ap.add_argument("--outdir", required=True, help="Dir to write combined files")
    ap.add_argument("--all-windows", action="store_true",
                    help="Also combine across all windows for this theory (run once).")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir).resolve()
    outdir = Path(args.outdir).resolve()
    combine_dir(frames_dir, outdir)

    if args.all_windows:
        # theory dir …/windows/<WIN>/wtp/<TAG>
        theory_dir = outdir
        wtp_dir = theory_dir.parent          # …/wtp
        tag = theory_dir.name
        win_root = wtp_dir.parent.parent     # …/windows
        # gather this theory across all windows
        arrays = []
        for tdir in sorted(win_root.glob(f"*/wtp/{tag}")):
            f = tdir / "wtp_window_combined.npz"
            if f.exists():
                arrays.append(np.load(f))
        if arrays:
            cat = lambda k: np.concatenate([z[k] for z in arrays], axis=0)
            out_all = win_root.parent / "wtp_all" / tag
            out_all.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                    out_all / "wtp_all_windows.npz",
                    frame_index=cat("frame_index"), E_QM=cat("E_QM"),
                    E_ref=cat("E_ref"), dE=cat("dE")
                    )
            # Forces optional
            if "forces" in arrays[0]:
                np.savez_compressed(
                        out_all / "forces_all_windows.npz",
                        forces=cat("forces")
                        )

if __name__ == "__main__":
    main()

