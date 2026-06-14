#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


N_WINDOWS = 42
N_FRAMES = 500
QM_INDEX = np.array([60, 53, 52], dtype=int) - 1
OUTPUT = Path("pos_all.npy")


def load_qmhub_positions(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing QMHub input file: {path}. This script expects visible "
            "qmhub/qmmm.inp_???? files and does not unpack qmhub.squashfs."
        )
    return np.loadtxt(path, usecols=(0, 1, 2), skiprows=1)[QM_INDEX]


def build_positions():
    pos_all = []
    for window in range(N_WINDOWS):
        window_positions = []
        input_dir = Path(f"../{window:02d}/qmhub")
        for frame in range(N_FRAMES):
            window_positions.append(load_qmhub_positions(input_dir / f"qmmm.inp_{frame:04d}"))
        pos_all.append(window_positions)
    return np.array(pos_all)


def main():
    parser = argparse.ArgumentParser(
        description="Build full pos_all.npy coordinates from per-window QMHub inputs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pos_all.npy.",
    )
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        raise FileExistsError(f"{OUTPUT} exists; rerun with --force to overwrite")

    np.save(OUTPUT, build_positions())


if __name__ == "__main__":
    main()
