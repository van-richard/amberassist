#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AMBER_HARTREE_TO_EV = 27.2114
EV_TO_KCAL = 23.061


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DFT TP energy differences against a reference method."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wb97xd_6-31+gd"],
        help="Method_basis labels to compare, for example wb97xd_6-31+gd.",
    )
    parser.add_argument(
        "--reference",
        default="b3lyp_6-31+gd",
        help="Reference method_basis label.",
    )
    parser.add_argument(
        "--reactant-window",
        type=int,
        default=0,
        help="Window index used as the reactant reference.",
    )
    parser.add_argument(
        "--ts-window",
        type=int,
        default=21,
        help="Window index used as the transition-state reference.",
    )
    parser.add_argument(
        "--energy-dir",
        type=Path,
        default=Path("../mbar/tp_energy"),
        help="Directory containing qmmm_<method_basis>_energy.npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dft_tp_deltaE.png"),
        help="Output plot filename.",
    )
    return parser.parse_args()


def load_delta_energy(energy_dir, method, reactant_window, ts_window):
    fname = energy_dir / f"qmmm_{method}_energy.npy"
    energy = np.load(fname)
    delta = np.array(energy[reactant_window] - energy[ts_window]) * AMBER_HARTREE_TO_EV
    return (delta - delta.mean()) * EV_TO_KCAL


def plot_methods(args):
    ref_energy = load_delta_energy(
        args.energy_dir, args.reference, args.reactant_window, args.ts_window
    )

    n_methods = len(args.methods)
    fig, axs = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), squeeze=False)

    for method, ax in zip(args.methods, axs.ravel()):
        method_energy = load_delta_energy(
            args.energy_dir, method, args.reactant_window, args.ts_window
        )
        ax.plot([-80, 80], [-80, 80], color="k", linewidth=1)
        ax.scatter(ref_energy, method_energy, marker=".")
        ax.set_xlabel(f"Ref {args.reference} Delta E (kcal/mol)")
        ax.set_ylabel(f"TP {method} Delta E (kcal/mol)")
        ax.set_title(method)
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(args.output)


def main():
    plot_methods(parse_args())


if __name__ == "__main__":
    main()
