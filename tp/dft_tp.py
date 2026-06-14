#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np

from qmhub import QMMM


num_qm_atoms = 73
num_prot_atoms = 27086
qm_index = np.array([59, 52, 51])


def normalize_basis(basis):
    aliases = {
        "6-31+g*": "6-31+gd",
        "6-31g*": "6-31gd",
    }
    return aliases.get(basis.lower(), basis)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one QMMM single-point energy/force calculation with QMHub."
    )
    parser.add_argument("fin", help="Input qmhub/qmmm.inp_???? file.")
    parser.add_argument("method", help="Q-Chem method, for example wb97xd.")
    parser.add_argument(
        "basis",
        help="Q-Chem basis, for example 6-31+gd. 6-31+g* and 6-31g* are accepted aliases.",
    )
    parser.add_argument("output_dir", help="Directory for this frame's outputs.")
    parser.add_argument(
        "output_stem",
        help="Output stem without _energy.npy or _forces.npy, for example qmmm_wb97xd_6-31+gd.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing complete energy/forces outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fin = Path(args.fin)
    method = args.method
    basis = normalize_basis(args.basis)
    output_dir = Path(args.output_dir)
    output_stem = args.output_stem

    if not fin.is_file():
        raise FileNotFoundError(f"Missing QMHub input file: {fin}")

    output_dir.mkdir(parents=True, exist_ok=True)
    energy_file = output_dir / f"{output_stem}_energy.npy"
    forces_file = output_dir / f"{output_stem}_forces.npy"

    complete = energy_file.exists() and forces_file.exists()
    partial = energy_file.exists() != forces_file.exists()
    if complete and not args.force:
        print(f"Skipping complete frame: {output_dir}")
        return
    if partial and not args.force:
        raise RuntimeError(
            "Found only one expected output. Re-run with --force after checking "
            f"{energy_file} and {forces_file}."
        )

    qmmm = QMMM(mode="text", driver="sander", cwd=output_dir)
    qmmm.io.cwd.mkdir(exist_ok=True)
    qmmm.setup_simulation()

    energy = []
    forces = []

    qmmm.load_system(fin)
    qmmm.build_model(switching_type="lrec", cutoff=10.0, swdist=None, pbc=True)
    qmmm.add_engine(
        "qchem",
        options={
            "method": method,
            "basis": basis,
            "scf_convergence": "9",
        },
    )

    energy.append(np.copy(qmmm.simulation.energy))
    forces.append(np.copy(qmmm.simulation.energy_gradient[:, qmmm.system.qm_index]))

    np.save(energy_file, energy)
    np.save(forces_file, forces)
    print(f"Finished {fin} with {method}/{basis}: {output_dir}")


if __name__ == "__main__":
    main()
