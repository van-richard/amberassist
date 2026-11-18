#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np

from qmhub import QMMM


def run_wtp_qchem(fin: Path, method: str, basis: str, want_grad: bool, cwd: Path):
    """Evaluate one frame with the legacy QMHub setup and frame-local Q-Chem files.

    Return the target energy, a zero reference placeholder, and optional positive
    QM-subset energy gradients under the historical ``forces`` key.
    """
    # Initialize the Sander text reader and MD simulation before loading a frame.
    qmmm = QMMM(mode="text", driver="sander", cwd=cwd)
    qmmm.io.cwd.mkdir(parents=True, exist_ok=True)
    qmmm.setup_simulation()
    qmmm.load_system(str(fin))
    # Build the same periodic LREC embedding used by the legacy worker.
    qmmm.build_model(switching_type="lrec", cutoff=10.0, swdist=None, pbc=True)

    # Match the legacy SCF threshold; additional Q-Chem settings are opt-in.
    options = {
        "method": method,
        "basis": basis,
        "scf_convergence": os.getenv("QCHEM_SCF_CONV", "9"),
    }
    for variable, keyword in (
        ("QCHEM_MEM", "mem_total"),
        ("QCHEM_MAX_SCF", "max_scf_cycles"),
        ("QCHEM_DFT_GRID", "dft_grid"),
    ):
        if variable in os.environ:
            options[keyword] = os.environ[variable]
    # Register Q-Chem with the model; result access below triggers evaluation.
    qmmm.add_engine("qchem", options=options)

    # QMHub evaluates lazily. Extract its one-element energy array as a scalar.
    energy = np.asarray(qmmm.simulation.energy).item()
    reference = 0.0  # Target-energy evaluation only: no reference model is run.
    res = {
        "E_QM": energy,
        "E_ref": reference,
        "dE": energy - reference,
        "meta": {
            "inp": fin.name,
            "engine": "qchem",
            "method": method,
            "basis": basis,
            "reference_energy_source": "zero_placeholder",
        },
    }
    if want_grad:
        # Keep positive gradients in (xyz, qm_atom) order, as in the legacy file.
        # Select QM columns in their declared order and retain the "forces" key.
        res["forces"] = np.array(
            qmmm.simulation.energy_gradient[:, qmmm.system.qm_index],
            dtype=np.float32,
        )
        res["meta"]["forces_quantity"] = "positive_qm_subset_energy_gradient"
        res["meta"]["forces_axes"] = ["xyz", "qm_atom"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Path to qmmm.inp_????")
    ap.add_argument("--outdir", default=".", help="Output directory (per-frame npz)")
    ap.add_argument("--method", default=os.getenv("WTP_METHOD", "B3LYP"))
    ap.add_argument("--basis", default=os.getenv("WTP_BASIS", "6-31+G(d)"))
    ap.add_argument(
        "--grad", action="store_true",
        help="Save legacy positive QM-subset energy gradients under the forces key",
    )
    args = ap.parse_args()

    # Reject invalid input before creating output directories or a QMHub backend.
    fin = Path(args.inp).resolve()
    if not fin.is_file():
        ap.error(f"Input file does not exist: {fin}")

    # Preserve the frame suffix, including zero padding, for the output filename.
    idx = "".join(ch for ch in fin.name.split("_")[-1] if ch.isdigit())
    if not idx:
        ap.error(f"Cannot parse frame index from {fin.name}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Parallel frames/theories must not share qchem.inp, output, or save files.
    cwd = outdir / "work" / fin.name
    rec = run_wtp_qchem(fin, args.method, args.basis, args.grad, cwd=cwd)

    # Save one compact record per frame; JSON metadata avoids pickled objects.
    save = {k: v for k, v in rec.items() if k != "meta"}
    save["meta"] = json.dumps(rec["meta"])
    np.savez_compressed(outdir / f"wtp_frame_{idx}.npz", **save)


if __name__ == "__main__":
    main()
