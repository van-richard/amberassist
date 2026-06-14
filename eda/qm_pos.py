#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np


N_WINDOWS = 42
N_FRAMES = 500
FRAME_STRIDE = 2
INPUT = Path("pos_all.npy")
OUTPUT = Path("qm_pos.npy")


def build_qm_positions():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing {INPUT}; run pos_all.py first")
    coords = np.load(INPUT)
    return coords[:N_WINDOWS, :N_FRAMES:FRAME_STRIDE, :, :]


def main():
    parser = argparse.ArgumentParser(
        description="Build truncated qm_pos.npy coordinates from pos_all.npy."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing qm_pos.npy.",
    )
    args = parser.parse_args()

    if OUTPUT.exists() and not args.force:
        raise FileExistsError(f"{OUTPUT} exists; rerun with --force to overwrite")

    np.save(OUTPUT, build_qm_positions())


if __name__ == "__main__":
    main()
