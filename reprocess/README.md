# QM/MM Reprocessing With QMHub

This folder contains the helper files used to re-evaluate saved umbrella-sampling
production trajectories with Amber `sander.MPI` and QMHub. The scripts are
workflow glue only; they should not change the scientific model, trajectory data,
or generated reference products.

## Files

| File | Purpose |
| --- | --- |
| `reprocess.slurm` | SLURM array wrapper for per-window reprocessing. |
| `write_mdin.sh` | Expands `step7_reprocess.mdin.tmp` into `step7_reprocess.mdin` inside a window directory. |
| `step7_reprocess.mdin.tmp` | Amber/QMHub single-point reprocessing template. |
| `qmhub2.ini` | QMHub configuration used by the Amber template. |
| `dedup.sh` | Optional cleanup for duplicate generated `qmmm.inp_*` files. |
| `legacy/` | Older notes/scripts kept for reference. |

## Workflow

`reprocess.slurm` expects a `list` file in the project root, with one umbrella
window per line, such as `00`, `01`, ..., `41`. Each SLURM array task reads one
window name, enters that window directory, links `qmhub2.ini`, generates
`step7_reprocess.mdin`, and runs:

```bash
sander.MPI -O \
  -i step7_reprocess.mdin \
  -o step7_reprocess.mdout \
  -p step3_pbcsetup.parm7 \
  -c step5.00_equilibration.ncrst \
  -y step6*.nc \
  -x step7_reprocess.nc
```

The current script uses per-window `step6*.nc` production trajectories. If a
future workflow should use the concatenated `step6_all.nc` trajectory instead,
update the script deliberately and document that change before running it.

`write_mdin.sh` first reads `../input/qm_info.txt` from the window directory.
That file should define `qmmask = ...` and `qmcharge = ...`; these values replace
`__QMMASK__` and `__QMCHARGE__` in the template. If `../input/qm_info.txt` is
absent, the script extracts those two metadata values from the current window's
`step6.00_equilibration.mdin`. The fallback reads the production mdin only; it
does not modify it. The QMHub base path resolves symlinks and points to the real
current window's `qmhub` directory so it can be archived later as
`qmhub.squashfs`.

## Generated Files

Reprocessing can create or update files inside each window directory, including:

* `step7_reprocess.mdin`
* `step7_reprocess.mdout`
* `step7_reprocess.nc`
* `qmhub/`
* `qmmm.inp_????` files inside the QMHub output directory

Treat `step6*.nc`, `step6_all.nc`, `qmhub.squashfs`, and any archived
`qmhub_inp.????` data products as read-only unless regeneration is explicitly
requested.

## Optional Deduplication

QMHub may generate duplicate `qmmm.inp_*` files. After a successful reprocessing
run, inspect the generated directory first:

```bash
bash ../reprocess/dedup.sh --dry-run qmhub
```

Then run without `--dry-run` only when the planned removals and renumbering are
expected:

```bash
bash ../reprocess/dedup.sh qmhub
```

`dedup.sh` refuses to overwrite existing destination filenames during renumbering.

## Safety Notes

Do not run these scripts from automation unless the target windows and inputs
have been checked. `reprocess.slurm` launches Amber/QMHub work and is not a
lightweight validation command.
