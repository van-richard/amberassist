# QMMM Input Workflow

Edit `write_info.sh` for the system, then run the setup scripts from this
directory:

```bash
bash write_info.sh
bash write_mdin.sh
bash gen_inputs.sh
bash gen_cvs.sh
```

`write_info.sh` writes `qm_info.txt`, copies the topology and restart from
`REF`, and stores the QM region, charge, thermostat, window count, and QMHub
mode used by later scripts.

For QMHub runs, set:

```bash
QMTHEORY="EXTERN"
QMHUB_MODE="DFT"  # uses qmhub_dft.ini
```

or:

```bash
QMTHEORY="EXTERN"
QMHUB_MODE="MTS"  # uses qmhub_mts.ini
```

For non-QMHub Amber QM/MM runs, set `QMTHEORY` to the desired Amber-supported
method. `gen_inputs.sh` only links `qmhub.ini` when `QMTHEORY="EXTERN"`.

`reprocess/write_mdin.sh` is a separate reprocessing helper for generating
`step7_reprocess.mdin` inside existing window directories; it is not step 2 of
this input-generation workflow.
