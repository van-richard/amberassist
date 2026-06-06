#!/usr/bin/env python
"""Generate project-specific AMBER CV restraints for an aldol-reaction workflow.

Review the atom masks and restraint distances before reusing this script for a
different molecular system.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence


# Label, atom mask 1, atom mask 2, state A distance, state B distance, final distance.
MASKS = (
    ("PA1", ":332&@NZ", ":471&@H6", 5.00, 1.00, 4.50),
    ("PL1", ":471&@O3", ":471&@H6", 1.00, 1.00, 5.00),
    ("A", ":471&@C6", ":471&@C7", 1.00, 1.00, 5.00),
    ("PA2", ":471&@N3", ":13574&@H2", 3.00, 1.00, 5.00),
    ("PL2", ":13574&@O", ":13574&@H2", 1.00, 3.50, 1.00),
)


def _linear_value(start: float, stop: float, index: int, count: int) -> float:
    return start + (stop - start) * index / (count - 1)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parm", type=Path, default=Path("step3_pbcsetup.parm7"))
    parser.add_argument("--coord", type=Path, default=Path("step3_pbcsetup.ncrst"))
    parser.add_argument("--n-windows", type=int, default=40)
    parser.add_argument("--output-pattern", default="../{window:02d}/cv.rst")
    parser.add_argument("--rk2", type=float, default=150.0)
    parser.add_argument("--rk3", type=float, default=150.0)
    parser.add_argument("--r1", type=float, default=0.0)
    parser.add_argument("--r4", type=float, default=10.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output files and restraints without writing files.",
    )
    return parser.parse_args(argv)


def restraint_value_for_window(
    label: str,
    distance1: float,
    distance2: float,
    distance3: float,
    window: int,
    n_windows: int,
) -> float:
    if n_windows < 2:
        raise ValueError("n_windows must be at least 2")
    if not 0 <= window < n_windows:
        raise ValueError(f"window must be between 0 and {n_windows - 1}")

    if distance1 == distance2 and label in {"PL1", "A"}:
        return _linear_value(distance1, distance3, window, n_windows)

    if distance1 != distance2 and label in {"PA1", "PA2", "PL2"}:
        first_stage_windows = n_windows // 2
        second_stage_windows = n_windows - first_stage_windows
        if first_stage_windows < 2 or second_stage_windows < 2:
            raise ValueError("n_windows must provide at least two windows per interpolation stage")
        if window < first_stage_windows:
            return _linear_value(distance1, distance2, window, first_stage_windows)
        return _linear_value(
            distance2,
            distance3,
            window - first_stage_windows,
            second_stage_windows,
        )

    raise ValueError(f"unsupported restraint definition for label {label!r}")


def select_single_atom(topology: object, mask: str) -> int:
    selected = topology.select(mask)
    count = len(selected)
    if count == 0:
        raise ValueError(f"atom mask {mask!r} selected zero atoms")
    if count > 1:
        raise ValueError(f"atom mask {mask!r} selected {count} atoms; expected exactly one")
    return int(selected[0]) + 1


def render_restraint_block(
    label: str,
    mask1: str,
    mask2: str,
    atom1: int,
    atom2: int,
    restraint_value: float,
    r1: float = 0.0,
    r4: float = 10.0,
    rk2: float = 150.0,
    rk3: float = 150.0,
) -> str:
    return (
        f"# {label} {mask1} {mask2}\n"
        " &rst\n"
        f"  iat={atom1},{atom2},\n"
        f"  r1={r1:g}, r2={restraint_value:.2f}, r3={restraint_value:.2f}, r4={r4:g},\n"
        f"  rk2={rk2}, rk3={rk3},\n"
        " &end\n"
    )


def output_path_for_window(output_pattern: str, window: int) -> Path:
    try:
        return Path(output_pattern.format(window=window))
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            f"invalid output pattern {output_pattern!r}; use a '{{window}}' field"
        ) from exc


def write_restraints(
    topology: object,
    n_windows: int,
    output_pattern: str,
    r1: float,
    r4: float,
    rk2: float,
    rk3: float,
    dry_run: bool = False,
) -> None:
    selected_atoms = {
        mask: select_single_atom(topology, mask)
        for _, mask1, mask2, _, _, _ in MASKS
        for mask in (mask1, mask2)
    }

    for window in range(n_windows):
        output_path = output_path_for_window(output_pattern, window)
        blocks = []
        for label, mask1, mask2, distance1, distance2, distance3 in MASKS:
            value = restraint_value_for_window(
                label,
                distance1,
                distance2,
                distance3,
                window,
                n_windows,
            )
            blocks.append(
                render_restraint_block(
                    label,
                    mask1,
                    mask2,
                    selected_atoms[mask1],
                    selected_atoms[mask2],
                    value,
                    r1=r1,
                    r4=r4,
                    rk2=rk2,
                    rk3=rk3,
                )
            )

        contents = "".join(blocks)
        if dry_run:
            print(f"--- {output_path} ---")
            print(contents, end="")
        else:
            output_path.write_text(contents)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if not args.parm.is_file():
        raise SystemExit(f"error: AMBER parameter/topology file not found: {args.parm}")
    if not args.coord.is_file():
        raise SystemExit(f"error: AMBER coordinate file not found: {args.coord}")

    try:
        import pytraj as pt
    except ImportError as exc:
        raise SystemExit("error: pytraj is required to load the AMBER topology and coordinates") from exc

    try:
        trajectory = pt.load(str(args.coord), str(args.parm))
        write_restraints(
            trajectory.top,
            args.n_windows,
            args.output_pattern,
            args.r1,
            args.r4,
            args.rk2,
            args.rk3,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
