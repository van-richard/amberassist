#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np

# --- QMHub imports (assumed available in your environment) ---
from qmhub import QMMM

def run_wtp_qchem(fin: Path, method: str, basis: str, want_grad: bool):
    """
    Evaluate QM energy (and optionally forces) for one qmmm.inp_???? using Q-Chem via QMHub.
    Returns a dict with {E_QM, E_ref, dE, forces?, meta}.
    """
    qmmm = QMMM.from_file(str(fin))

    # Configure Q-Chem engine (match your prior configuration; adjust keys as needed)
    # NOTE: QMHub's API specifics may vary slightly by version.
    qmmm.set_qm_engine(
            engine="qchem",
            options={
                "method": method,
                "basis": basis,
                "mem_total": os.getenv("QCHEM_MEM", "4000"),         # MB
                "scf_convergence": os.getenv("QCHEM_SCF_CONV", "8"), # 10^-8
                "max_scf_cycles": int(os.getenv("QCHEM_MAX_SCF", "200")),
                "dft_grid": os.getenv("QCHEM_DFT_GRID", "SG-1"),
                # Add dispersion or special flags via env if desired:
                # "d3": os.getenv("QCHEM_D3", "false"),
                }
            )

    qmmm.run(gradients=want_grad)

    # Energies: adapt if you maintain a separate reference (MM or semiempirical)
    E_qm = float(qmmm.simulation.energy)
    E_ref = 0.0  # Replace if you have a meaningful reference energy per frame
    dE = E_qm - E_ref

    res = {"E_QM": E_qm, "E_ref": E_ref, "dE": dE,
           "meta": {"inp": fin.name, "engine": "qchem", "method": method, "basis": basis}}
    if want_grad and hasattr(qmmm.simulation, "forces") and qmmm.simulation.forces is not None:
        # forces on QM atoms; store float32 to keep files small
        res["forces"] = np.array(qmmm.simulation.forces, dtype=np.float32)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Path to qmmm.inp_????")
    ap.add_argument("--outdir", default=".", help="Output directory (per-frame npz)")
    ap.add_argument("--method", default=os.getenv("WTP_METHOD", "B3LYP"))
    ap.add_argument("--basis", default=os.getenv("WTP_BASIS", "6-31+G(d)"))
    ap.add_argument("--grad", action="store_true", help="Also compute/store forces")
    args = ap.parse_args()

    fin = Path(args.inp).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # parse frame index (qmmm.inp_0007 → 0007)
    idx = "".join(ch for ch in fin.name.split("_")[-1] if ch.isdigit())
    if not idx:
        raise SystemExit(f"Cannot parse frame index from {fin.name}")

    rec = run_wtp_qchem(fin, args.method, args.basis, args.grad)

    # Save compact per-frame NPZ
    save = {k: v for k, v in rec.items() if k != "meta"}
    save["meta"] = json.dumps(rec["meta"])
    np.savez_compressed(outdir / f"wtp_frame_{idx}.npz", **save)

if __name__ == "__main__":
    main()

