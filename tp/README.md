# DFT Thermodynamic Perturbation Workflow

This folder contains helper scripts for DFT single-point QM/MM thermodynamic
perturbation (TP) calculations on umbrella-window configurations. The workflow
reads existing QMHub text inputs from the window folders, evaluates energies and
forces with Q-Chem through QMHub, combines frame/window energies, and provides a
lightweight plotting script for comparing TP energy differences.

The scripts do not unpack `qmhub.squashfs` and do not regenerate QMHub inputs.
Run `dft_tp.slurm` only after the required window has an unpacked
`qmhub/qmmm.inp_????` directory.

## Directory Tree

```text
tp/
├── README.md
├── dft_tp.py
├── dft_tp.slurm
├── combine_dft_tp.py
├── combine_dft_tp.ipynb
├── plot_tp.py
└── plot_tp.ipynb
```

`dft_tp.py` runs one QMHub/Q-Chem single-point calculation for one
`qmmm.inp_????` file. It writes one energy array and one forces array for that
frame.

`dft_tp.slurm` is the Slurm array driver. It selects umbrella windows from
`../list`, loops over frame indices, skips complete outputs by default, and
passes the selected DFT method and basis to `dft_tp.py`.

`combine_dft_tp.py` and `combine_dft_tp.ipynb` combine frame-level energies from
all windows for one or more `method_basis` labels. Combined MBAR-facing arrays
are written to `../mbar/tp_energy/`.

`plot_tp.py` and `plot_tp.ipynb` plot TP energy differences from combined
`../mbar/tp_energy/qmmm_<method_basis>_energy.npy` files.

## Inputs And Outputs

Expected per-window input:

```text
../<window>/qmhub/qmmm.inp_????
```

Frame-level DFT TP outputs:

```text
tp/qmmm_energies/<method>_<basis>/<window>/<qmmm_index>/
├── qmmm_<method>_<basis>_energy.npy
└── qmmm_<method>_<basis>_forces.npy
```

Combined energy outputs:

```text
tp/qmmm_energies/<method>_<basis>/<window>/qmmm_<method>_<basis>_energy_all.npy
../mbar/tp_energy/qmmm_<method>_<basis>_energy.npy
```

Basis aliases are normalized for output naming and Q-Chem options:

```text
6-31+g* -> 6-31+gd
6-31g*  -> 6-31gd
```

## Example Workflow

Edit the run controls near the top of `dft_tp.slurm`:

```bash
method="wb97xd"
basis="6-31+gd"
strt="0"
end="498"
iter="2"
overwrite="0"
```

Submit from the repository root:

```bash
sbatch tp/dft_tp.slurm
```

Or from this `tp` directory:

```bash
sbatch dft_tp.slurm
```

Use `overwrite="1"` only after checking existing outputs. With
`overwrite="0"`, complete frame outputs are skipped and partial frame outputs
cause the job to stop.

After the frame jobs finish, edit `methods` in `combine_dft_tp.py` or
`combine_dft_tp.ipynb`, then run:

```bash
cd tp
python combine_dft_tp.py
```

To plot combined TP energy differences:

```bash
cd tp
python plot_tp.py \
  --reference b3lyp_6-31+gd \
  --methods wb97xd_6-31+gd \
  --reactant-window 0 \
  --ts-window 21
```

## Safety Notes

The `00` through `41` window folders are reference simulation data and may be
symlinks. These scripts read `../<window>/qmhub/qmmm.inp_????` but write new TP
results under `tp/qmmm_energies/`.

Do not unpack, overwrite, or regenerate `qmhub.squashfs` as part of this TP
workflow. If `dft_tp.slurm` reports a missing `qmhub/qmmm.inp_????`, unpack or
mount the QMHub inputs outside this workflow before resubmitting.
