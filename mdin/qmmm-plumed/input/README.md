# QMMM + PLUMED Input Workflow

Edit `write_info.sh` for the system, then run these scripts from this
directory:

```bash
bash write_info.sh
bash write_mdin.sh
bash gen_plumeddat.sh
```

The generated runtime files are written one directory up, beside
`pete_runqmmm.slurm`:

- `step5.00_equilibration.mdin`
- `plumed.dat`
- `step3_pbcsetup.parm7`
- `step5.00_equilibration_inp.ncrst`
- `qmhub.ini` when `QMTHEORY="EXTERN"`

Before running `write_info.sh`, set `REF` to a directory containing
`step3_pbcsetup.parm7` and `prod00.ncrst`.

## QM/MM Settings

Use `QMTHEORY="EXTERN"` for QMHub. Set:

```bash
QMHUB_MODE="DFT"
```

to use `qmhub_dft.ini`, or:

```bash
QMHUB_MODE="MTS"
```

to use `qmhub_mts.ini`.

For non-QMHub Amber QM/MM, set `QMTHEORY` to the desired Amber-supported method.
In that case `qmhub.ini` is not created.

## QM Region

The default QM region is built from named components in `write_info.sh`:

- `PROTEIN_RESIDUES`
- `METAL_RESIDUES`
- `WATER_RESIDUES`
- `NA_MASK`

For full manual control, set `QMMASK_OVERRIDE` to an explicit Amber mask.

## PLUMED Template

Choose the PLUMED template with:

```bash
PLUMED_METHOD="metad"   # metad or wtmetad
PLUMED_CV="2d"          # 2d or d1-d2
```

The script selects `plumed.${PLUMED_METHOD}.${PLUMED_CV}.dat` and writes it to
`../plumed.dat`. Add a matching template file before selecting a new
method/CV combination.
