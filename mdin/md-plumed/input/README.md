# Classical MD + PLUMED Input Workflow

Edit `write_info.sh` for the system, then run these scripts from this
directory:

```bash
bash write_info.sh
bash write_mdin.sh
bash gen_plumeddat.sh
```

The generated runtime files are written one directory up, beside
`pete_runmd.slurm`:

- `step5.00_equilibration.mdin`
- `plumed.dat`
- `step3_pbcsetup.parm7`
- `step5.00_equilibration_inp.ncrst`

Before running `write_info.sh`, set `REF` to a directory containing
`step3_pbcsetup.parm7` and `prod00.ncrst`.

## MD Settings

Edit these values in `write_info.sh`:

- `THERMOSTAT`: `langevin` or `sinr`
- `NSTEPS`: number of MD steps
- `PRINTFREQ`: energy and coordinate output frequency

## PLUMED Template

Choose the PLUMED template with:

```bash
PLUMED_METHOD="metad"   # metad or wtmetad
PLUMED_CV="2d"          # 2d or d1-d2
```

The script selects `plumed.${PLUMED_METHOD}.${PLUMED_CV}.dat` and writes it to
`../plumed.dat`. Add a matching template file before selecting a new
method/CV combination.

This workflow is for classical MD only. It does not create QMHub inputs,
`qmhub.ini`, Q-Chem settings, `&qmmm`, or `&qmhub` blocks.
