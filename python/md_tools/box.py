# md_tools/cli.py

import argparse
import glob
import os

from md_tools.topology import find_parm
from .workflows import print_box_info


def main():

    parser = argparse.ArgumentParser(
            description="MD trajectory frame counter"
            )

    parser.add_argument(
            "-p", "--parm",
            help="Topology file (parm7). If omitted, auto-detected."
            )

    parser.add_argument(
            "-y", "--traj",
            nargs="+",
            required=True,
            help="Trajectory file(s) or wildcard pattern(s)"
            )


    args = parser.parse_args()

    if args.parm:
        parm_file = args.parm
    else:
        parm_file = find_parm()

    # Expand wildcards manually
    traj_files = []
    for pattern in args.traj:
        expanded = glob.glob(pattern)
        if not expanded:
            raise RuntimeError(f"No files matched pattern: {pattern}")
        traj_files.extend(sorted(expanded))

    # Remove duplicates while preserving order
    traj_files = list(dict.fromkeys(traj_files))

    results = print_box_info(traj_files, parm_file)

    print()


if __name__ == "__main__":
    main()

