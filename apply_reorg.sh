#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"
echo 'Applying amberassist reorganization...'
mkdir -p "ambertools"
mkdir -p "ambertools/cpptraj"
mkdir -p "ambertools/cpptraj/notebooks"
mkdir -p "ambertools/tleap"
mkdir -p "io"
mkdir -p "io/slurm"
mkdir -p "mbar/notebooks"
mkdir -p "mdin/qmmm/asm/input"
mkdir -p "mdin/qmmm/mts/sinr/input"
mkdir -p "mdin/qmmm/semiempirical/param/input"
mkdir -p "mdin/qmmm/semiempirical/reparam"
mkdir -p "mlp/torchmdnet"
mkdir -p "notebooks"
if [ -e "mdin/qmmm/asm/reorgnize_ncrst2.py" ]; then git mv -f "mdin/qmmm/asm/reorgnize_ncrst2.py" "io/reorgnize_ncrst2.py" 2>/dev/null || mv -f "mdin/qmmm/asm/reorgnize_ncrst2.py" "io/reorgnize_ncrst2.py"; fi
if [ -e "mdin/qmmm/asm/sander3.slurm" ]; then git mv -f "mdin/qmmm/asm/sander3.slurm" "io/slurm/sander3.slurm" 2>/dev/null || mv -f "mdin/qmmm/asm/sander3.slurm" "io/slurm/sander3.slurm"; fi
if [ -e "mdin/qmmm/asm/runasm.slurm" ]; then git mv -f "mdin/qmmm/asm/runasm.slurm" "io/slurm/runasm.slurm" 2>/dev/null || mv -f "mdin/qmmm/asm/runasm.slurm" "io/slurm/runasm.slurm"; fi
if [ -e "mdin/qmmm/asm/runqmmm.slurm" ]; then git mv -f "mdin/qmmm/asm/runqmmm.slurm" "io/slurm/runqmmm.slurm" 2>/dev/null || mv -f "mdin/qmmm/asm/runqmmm.slurm" "io/slurm/runqmmm.slurm"; fi
if [ -e "mdin/qmmm/asm/runqmmm1.sh" ]; then git mv -f "mdin/qmmm/asm/runqmmm1.sh" "io/slurm/runqmmm1.sh" 2>/dev/null || mv -f "mdin/qmmm/asm/runqmmm1.sh" "io/slurm/runqmmm1.sh"; fi
if [ -e "mdin/qmmm/asm/run_colvar.sh" ]; then git mv -f "mdin/qmmm/asm/run_colvar.sh" "io/slurm/run_colvar.sh" 2>/dev/null || mv -f "mdin/qmmm/asm/run_colvar.sh" "io/slurm/run_colvar.sh"; fi
if [ -e "mdin/qmmm/asm/reorgnize_ncrst.py" ]; then git mv -f "mdin/qmmm/asm/reorgnize_ncrst.py" "io/reorgnize_ncrst.py" 2>/dev/null || mv -f "mdin/qmmm/asm/reorgnize_ncrst.py" "io/reorgnize_ncrst.py"; fi
if [ -e "mdin/qmmm/asm/sander-qchem.slurm" ]; then git mv -f "mdin/qmmm/asm/sander-qchem.slurm" "io/slurm/sander-qchem.slurm" 2>/dev/null || mv -f "mdin/qmmm/asm/sander-qchem.slurm" "io/slurm/sander-qchem.slurm"; fi
if [ -e "mdin/qmmm/asm/gen_groupfile.sh" ]; then git mv -f "mdin/qmmm/asm/gen_groupfile.sh" "io/gen_groupfile.sh" 2>/dev/null || mv -f "mdin/qmmm/asm/gen_groupfile.sh" "io/gen_groupfile.sh"; fi
if [ -e "mdin/qmmm/asm/gen_input.sh" ]; then git mv -f "mdin/qmmm/asm/gen_input.sh" "io/gen_input.sh" 2>/dev/null || mv -f "mdin/qmmm/asm/gen_input.sh" "io/gen_input.sh"; fi
if [ -e "mdin/qmmm/asm/runasm2.slurm" ]; then git mv -f "mdin/qmmm/asm/runasm2.slurm" "io/slurm/runasm2.slurm" 2>/dev/null || mv -f "mdin/qmmm/asm/runasm2.slurm" "io/slurm/runasm2.slurm"; fi
if [ -e "mdin/qmmm/asm/gen_groupfile2.sh" ]; then git mv -f "mdin/qmmm/asm/gen_groupfile2.sh" "io/gen_groupfile2.sh" 2>/dev/null || mv -f "mdin/qmmm/asm/gen_groupfile2.sh" "io/gen_groupfile2.sh"; fi
if [ -e "mdin/qmmm/mts/langevin/runqmmm1.sh" ]; then git mv -f "mdin/qmmm/mts/langevin/runqmmm1.sh" "io/slurm/runqmmm1.sh" 2>/dev/null || mv -f "mdin/qmmm/mts/langevin/runqmmm1.sh" "io/slurm/runqmmm1.sh"; fi
if [ -e "mdin/qmmm/mts/langevin/input/gen_inputs.sh" ]; then git mv -f "mdin/qmmm/mts/langevin/input/gen_inputs.sh" "io/gen_inputs.sh" 2>/dev/null || mv -f "mdin/qmmm/mts/langevin/input/gen_inputs.sh" "io/gen_inputs.sh"; fi
if [ -e "mdin/qmmm/mts/langevin/input/gen_cv.sh" ]; then git mv -f "mdin/qmmm/mts/langevin/input/gen_cv.sh" "io/gen_cv.sh" 2>/dev/null || mv -f "mdin/qmmm/mts/langevin/input/gen_cv.sh" "io/gen_cv.sh"; fi
if [ -e "mdin/qmmm/mts/sinr/runqmmm1.sh" ]; then git mv -f "mdin/qmmm/mts/sinr/runqmmm1.sh" "io/slurm/runqmmm1.sh" 2>/dev/null || mv -f "mdin/qmmm/mts/sinr/runqmmm1.sh" "io/slurm/runqmmm1.sh"; fi
if [ -e "mdin/qmmm/reprocessing/step7_reprocessing.slurm" ]; then git mv -f "mdin/qmmm/reprocessing/step7_reprocessing.slurm" "io/slurm/step7_reprocessing.slurm" 2>/dev/null || mv -f "mdin/qmmm/reprocessing/step7_reprocessing.slurm" "io/slurm/step7_reprocessing.slurm"; fi
if [ -e "mdin/qmmm/fmatch/training_set_torch.py" ]; then git mv -f "mdin/qmmm/fmatch/training_set_torch.py" "io/training_set_torch.py" 2>/dev/null || mv -f "mdin/qmmm/fmatch/training_set_torch.py" "io/training_set_torch.py"; fi
if [ -e "mdin/qmmm/fmatch/list.training_set" ]; then git mv -f "mdin/qmmm/fmatch/list.training_set" "io/list.training_set" 2>/dev/null || mv -f "mdin/qmmm/fmatch/list.training_set" "io/list.training_set"; fi
if [ -e "mdin/qmmm/fmatch/scatter.ipynb" ]; then git mv -f "mdin/qmmm/fmatch/scatter.ipynb" "notebooks/scatter.ipynb" 2>/dev/null || mv -f "mdin/qmmm/fmatch/scatter.ipynb" "notebooks/scatter.ipynb"; fi
if [ -e "mdin/qmmm/fmatch/plot.ipynb" ]; then git mv -f "mdin/qmmm/fmatch/plot.ipynb" "notebooks/plot.ipynb" 2>/dev/null || mv -f "mdin/qmmm/fmatch/plot.ipynb" "notebooks/plot.ipynb"; fi
if [ -e "mdin/qmmm/fmatch/training_set.py" ]; then git mv -f "mdin/qmmm/fmatch/training_set.py" "io/training_set.py" 2>/dev/null || mv -f "mdin/qmmm/fmatch/training_set.py" "io/training_set.py"; fi
if [ -e "mdin/qmmm/fmatch/training_set_model.py" ]; then git mv -f "mdin/qmmm/fmatch/training_set_model.py" "io/training_set_model.py" 2>/dev/null || mv -f "mdin/qmmm/fmatch/training_set_model.py" "io/training_set_model.py"; fi
if [ -e "mdin/qmmm/fmatch/checkframes.ipynb" ]; then git mv -f "mdin/qmmm/fmatch/checkframes.ipynb" "notebooks/checkframes.ipynb" 2>/dev/null || mv -f "mdin/qmmm/fmatch/checkframes.ipynb" "notebooks/checkframes.ipynb"; fi
if [ -e "mdin/qmmm/semiempirical/param/runqmmm.slurm" ]; then git mv -f "mdin/qmmm/semiempirical/param/runqmmm.slurm" "io/slurm/runqmmm.slurm" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/runqmmm.slurm" "io/slurm/runqmmm.slurm"; fi
if [ -e "mdin/qmmm/semiempirical/param/runqmmm1.sh" ]; then git mv -f "mdin/qmmm/semiempirical/param/runqmmm1.sh" "io/slurm/runqmmm1.sh" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/runqmmm1.sh" "io/slurm/runqmmm1.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/gen_inputs.sh" ]; then git mv -f "mdin/qmmm/semiempirical/param/input/gen_inputs.sh" "io/gen_inputs.sh" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/input/gen_inputs.sh" "io/gen_inputs.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/gen_qmmask.py" ]; then git mv -f "mdin/qmmm/semiempirical/param/input/gen_qmmask.py" "io/gen_qmmask.py" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/input/gen_qmmask.py" "io/gen_qmmask.py"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/tleap.sh" ]; then git mv -f "mdin/qmmm/semiempirical/param/input/tleap.sh" "ambertools/tleap/tleap.sh" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/input/tleap.sh" "ambertools/tleap/tleap.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/gen_cvs.py" ]; then git mv -f "mdin/qmmm/semiempirical/param/input/gen_cvs.py" "io/gen_cvs.py" 2>/dev/null || mv -f "mdin/qmmm/semiempirical/param/input/gen_cvs.py" "io/gen_cvs.py"; fi
if [ -e "mdin/md/runprod.slurm" ]; then git mv -f "mdin/md/runprod.slurm" "io/slurm/runprod.slurm" 2>/dev/null || mv -f "mdin/md/runprod.slurm" "io/slurm/runprod.slurm"; fi
if [ -e "mdin/md/runmin.slurm" ]; then git mv -f "mdin/md/runmin.slurm" "io/slurm/runmin.slurm" 2>/dev/null || mv -f "mdin/md/runmin.slurm" "io/slurm/runmin.slurm"; fi
if [ -e "mdin/md/runmd.slurm" ]; then git mv -f "mdin/md/runmd.slurm" "io/slurm/runmd.slurm" 2>/dev/null || mv -f "mdin/md/runmd.slurm" "io/slurm/runmd.slurm"; fi
if [ -e "mbar/get_coordination.ipynb" ]; then git mv -f "mbar/get_coordination.ipynb" "mbar/notebooks/get_coordination.ipynb" 2>/dev/null || mv -f "mbar/get_coordination.ipynb" "mbar/notebooks/get_coordination.ipynb"; fi
if [ -e "mbar/mbar-rolling.ipynb" ]; then git mv -f "mbar/mbar-rolling.ipynb" "mbar/notebooks/mbar-rolling.ipynb" 2>/dev/null || mv -f "mbar/mbar-rolling.ipynb" "mbar/notebooks/mbar-rolling.ipynb"; fi
if [ -e "mbar/get_rxn.ipynb" ]; then git mv -f "mbar/get_rxn.ipynb" "mbar/notebooks/get_rxn.ipynb" 2>/dev/null || mv -f "mbar/get_rxn.ipynb" "mbar/notebooks/get_rxn.ipynb"; fi
if [ -e "mbar/get_data.ipynb" ]; then git mv -f "mbar/get_data.ipynb" "mbar/notebooks/get_data.ipynb" 2>/dev/null || mv -f "mbar/get_data.ipynb" "mbar/notebooks/get_data.ipynb"; fi
if [ -e "mbar/mbar.ipynb" ]; then git mv -f "mbar/mbar.ipynb" "mbar/notebooks/mbar.ipynb" 2>/dev/null || mv -f "mbar/mbar.ipynb" "mbar/notebooks/mbar.ipynb"; fi
if [ -e "mbar/mbar0.ipynb" ]; then git mv -f "mbar/mbar0.ipynb" "mbar/notebooks/mbar0.ipynb" 2>/dev/null || mv -f "mbar/mbar0.ipynb" "mbar/notebooks/mbar0.ipynb"; fi
if [ -e "mbar/distances.ipynb" ]; then git mv -f "mbar/distances.ipynb" "mbar/notebooks/distances.ipynb" 2>/dev/null || mv -f "mbar/distances.ipynb" "mbar/notebooks/distances.ipynb"; fi
if [ -e "_drafts/bash/gen_ligand.sh" ]; then git mv -f "_drafts/bash/gen_ligand.sh" "io/gen_ligand.sh" 2>/dev/null || mv -f "_drafts/bash/gen_ligand.sh" "io/gen_ligand.sh"; fi
if [ -e "ambertools/tleap.ligand.sh" ]; then git mv -f "ambertools/tleap.ligand.sh" "ambertools/tleap/tleap.ligand.sh" 2>/dev/null || mv -f "ambertools/tleap.ligand.sh" "ambertools/tleap/tleap.ligand.sh"; fi
if [ -e "ambertools/cpptraj.pca2.sh" ]; then git mv -f "ambertools/cpptraj.pca2.sh" "ambertools/cpptraj/cpptraj.pca2.sh" 2>/dev/null || mv -f "ambertools/cpptraj.pca2.sh" "ambertools/cpptraj/cpptraj.pca2.sh"; fi
if [ -e "ambertools/tleap.126.sh" ]; then git mv -f "ambertools/tleap.126.sh" "ambertools/tleap/tleap.126.sh" 2>/dev/null || mv -f "ambertools/tleap.126.sh" "ambertools/tleap/tleap.126.sh"; fi
if [ -e "ambertools/parmed.1264.sh" ]; then git mv -f "ambertools/parmed.1264.sh" "ambertools/parmed.sh" 2>/dev/null || mv -f "ambertools/parmed.1264.sh" "ambertools/parmed.sh"; fi
if [ -e "ambertools/tleap.setbox.sh" ]; then git mv -f "ambertools/tleap.setbox.sh" "ambertools/tleap/tleap.setbox.sh" 2>/dev/null || mv -f "ambertools/tleap.setbox.sh" "ambertools/tleap/tleap.setbox.sh"; fi
if [ -e "ambertools/tleap.1264.sh" ]; then git mv -f "ambertools/tleap.1264.sh" "ambertools/tleap/tleap.1264.sh" 2>/dev/null || mv -f "ambertools/tleap.1264.sh" "ambertools/tleap/tleap.1264.sh"; fi
if [ -e "ambertools/cpptraj.pca1.sh" ]; then git mv -f "ambertools/cpptraj.pca1.sh" "ambertools/cpptraj/cpptraj.pca1.sh" 2>/dev/null || mv -f "ambertools/cpptraj.pca1.sh" "ambertools/cpptraj/cpptraj.pca1.sh"; fi
if [ -e "ambertools/pytraj/rmsd.ipynb" ]; then git mv -f "ambertools/pytraj/rmsd.ipynb" "ambertools/cpptraj/notebooks/rmsd.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/rmsd.ipynb" "ambertools/cpptraj/notebooks/rmsd.ipynb"; fi
if [ -e "ambertools/pytraj/rmsd-pairwise.ipynb" ]; then git mv -f "ambertools/pytraj/rmsd-pairwise.ipynb" "ambertools/cpptraj/notebooks/rmsd-pairwise.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/rmsd-pairwise.ipynb" "ambertools/cpptraj/notebooks/rmsd-pairwise.ipynb"; fi
if [ -e "ambertools/pytraj/rmsf.ipynb" ]; then git mv -f "ambertools/pytraj/rmsf.ipynb" "ambertools/cpptraj/notebooks/rmsf.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/rmsf.ipynb" "ambertools/cpptraj/notebooks/rmsf.ipynb"; fi
if [ -e "ambertools/pytraj/rmsd.py" ]; then git mv -f "ambertools/pytraj/rmsd.py" "ambertools/cpptraj/rmsd.py" 2>/dev/null || mv -f "ambertools/pytraj/rmsd.py" "ambertools/cpptraj/rmsd.py"; fi
if [ -e "ambertools/pytraj/distances.ipynb" ]; then git mv -f "ambertools/pytraj/distances.ipynb" "ambertools/cpptraj/notebooks/distances.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/distances.ipynb" "ambertools/cpptraj/notebooks/distances.ipynb"; fi
if [ -e "ambertools/pytraj/rmsf.py" ]; then git mv -f "ambertools/pytraj/rmsf.py" "ambertools/cpptraj/rmsf.py" 2>/dev/null || mv -f "ambertools/pytraj/rmsf.py" "ambertools/cpptraj/rmsf.py"; fi
if [ -e "ambertools/pytraj/pca.py" ]; then git mv -f "ambertools/pytraj/pca.py" "ambertools/cpptraj/pca.py" 2>/dev/null || mv -f "ambertools/pytraj/pca.py" "ambertools/cpptraj/pca.py"; fi
if [ -e "ambertools/pytraj/pca.ipynb" ]; then git mv -f "ambertools/pytraj/pca.ipynb" "ambertools/cpptraj/notebooks/pca.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/pca.ipynb" "ambertools/cpptraj/notebooks/pca.ipynb"; fi
if [ -e "ambertools/pytraj/pca-Copy2.ipynb" ]; then git mv -f "ambertools/pytraj/pca-Copy2.ipynb" "ambertools/cpptraj/notebooks/pca-Copy2.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/pca-Copy2.ipynb" "ambertools/cpptraj/notebooks/pca-Copy2.ipynb"; fi
if [ -e "ambertools/pytraj/pca2.ipynb" ]; then git mv -f "ambertools/pytraj/pca2.ipynb" "ambertools/cpptraj/notebooks/pca2.ipynb" 2>/dev/null || mv -f "ambertools/pytraj/pca2.ipynb" "ambertools/cpptraj/notebooks/pca2.ipynb"; fi
if [ -e "ambertools/pytraj/nframes.py" ]; then git mv -f "ambertools/pytraj/nframes.py" "ambertools/cpptraj/nframes.py" 2>/dev/null || mv -f "ambertools/pytraj/nframes.py" "ambertools/cpptraj/nframes.py"; fi
if [ -e "ambertools/pytraj/gen_cvs.py" ]; then git mv -f "ambertools/pytraj/gen_cvs.py" "io/gen_cvs.py" 2>/dev/null || mv -f "ambertools/pytraj/gen_cvs.py" "io/gen_cvs.py"; fi
if [ -e "ambertools/pytraj/rmsd-pairwise.py" ]; then git mv -f "ambertools/pytraj/rmsd-pairwise.py" "ambertools/cpptraj/rmsd-pairwise.py" 2>/dev/null || mv -f "ambertools/pytraj/rmsd-pairwise.py" "ambertools/cpptraj/rmsd-pairwise.py"; fi
if [ -e "mlp/delta/sqm_training_set.slurm" ]; then git mv -f "mlp/delta/sqm_training_set.slurm" "io/slurm/sqm_training_set.slurm" 2>/dev/null || mv -f "mlp/delta/sqm_training_set.slurm" "io/slurm/sqm_training_set.slurm"; fi
if [ -e "mlp/delta/combine.slurm" ]; then git mv -f "mlp/delta/combine.slurm" "io/slurm/combine.slurm" 2>/dev/null || mv -f "mlp/delta/combine.slurm" "io/slurm/combine.slurm"; fi
if [ -e "mlp/delta/ml_qmmm_diff.ipynb" ]; then git mv -f "mlp/delta/ml_qmmm_diff.ipynb" "notebooks/ml_qmmm_diff.ipynb" 2>/dev/null || mv -f "mlp/delta/ml_qmmm_diff.ipynb" "notebooks/ml_qmmm_diff.ipynb"; fi
if [ -e "mlp/delta/reprocessing.slurm" ]; then git mv -f "mlp/delta/reprocessing.slurm" "io/slurm/reprocessing.slurm" 2>/dev/null || mv -f "mlp/delta/reprocessing.slurm" "io/slurm/reprocessing.slurm"; fi
if [ -e "mlp/delta/training_set1.slurm" ]; then git mv -f "mlp/delta/training_set1.slurm" "io/slurm/training_set1.slurm" 2>/dev/null || mv -f "mlp/delta/training_set1.slurm" "io/slurm/training_set1.slurm"; fi
if [ -e "mlp/delta/ml_qmmm_diff.slurm" ]; then git mv -f "mlp/delta/ml_qmmm_diff.slurm" "io/slurm/ml_qmmm_diff.slurm" 2>/dev/null || mv -f "mlp/delta/ml_qmmm_diff.slurm" "io/slurm/ml_qmmm_diff.slurm"; fi
if [ -e "mlp/delta/forces.slurm" ]; then git mv -f "mlp/delta/forces.slurm" "io/slurm/forces.slurm" 2>/dev/null || mv -f "mlp/delta/forces.slurm" "io/slurm/forces.slurm"; fi
if [ -e "mlp/delta/training_set1.py" ]; then git mv -f "mlp/delta/training_set1.py" "io/training_set1.py" 2>/dev/null || mv -f "mlp/delta/training_set1.py" "io/training_set1.py"; fi
if [ -e "mlp/delta/sqm_training_set.py" ]; then git mv -f "mlp/delta/sqm_training_set.py" "io/sqm_training_set.py" 2>/dev/null || mv -f "mlp/delta/sqm_training_set.py" "io/sqm_training_set.py"; fi
if [ -e "mdin/qmmm/asm/input/qmhub.ini" ]; then git rm -f "mdin/qmmm/asm/input/qmhub.ini" 2>/dev/null || rm -f "mdin/qmmm/asm/input/qmhub.ini"; fi
if [ -e "mdin/qmmm/mts/sinr/input/qmhub.ini" ]; then git rm -f "mdin/qmmm/mts/sinr/input/qmhub.ini" 2>/dev/null || rm -f "mdin/qmmm/mts/sinr/input/qmhub.ini"; fi
if [ -e "mdin/qmmm/mts/sinr/input/gen_inputs.sh" ]; then git rm -f "mdin/qmmm/mts/sinr/input/gen_inputs.sh" 2>/dev/null || rm -f "mdin/qmmm/mts/sinr/input/gen_inputs.sh"; fi
if [ -e "mdin/qmmm/mts/sinr/input/gen_cv.sh" ]; then git rm -f "mdin/qmmm/mts/sinr/input/gen_cv.sh" 2>/dev/null || rm -f "mdin/qmmm/mts/sinr/input/gen_cv.sh"; fi
if [ -e "mdin/qmmm/mts/sinr/input/cv.rst" ]; then git rm -f "mdin/qmmm/mts/sinr/input/cv.rst" 2>/dev/null || rm -f "mdin/qmmm/mts/sinr/input/cv.rst"; fi
if [ -e "mdin/qmmm/semiempirical/reparam/qmhub.ini" ]; then git rm -f "mdin/qmmm/semiempirical/reparam/qmhub.ini" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/reparam/qmhub.ini"; fi
if [ -e "mdin/qmmm/semiempirical/reparam/step5.00_equilibration.mdin" ]; then git rm -f "mdin/qmmm/semiempirical/reparam/step5.00_equilibration.mdin" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/reparam/step5.00_equilibration.mdin"; fi
if [ -e "mdin/qmmm/semiempirical/reparam/cv.rst" ]; then git rm -f "mdin/qmmm/semiempirical/reparam/cv.rst" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/reparam/cv.rst"; fi
if [ -e "mdin/qmmm/semiempirical/param/run1.sh" ]; then git rm -f "mdin/qmmm/semiempirical/param/run1.sh" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/param/run1.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/parmed.sh" ]; then git rm -f "mdin/qmmm/semiempirical/param/input/parmed.sh" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/param/input/parmed.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/gen_cv.sh" ]; then git rm -f "mdin/qmmm/semiempirical/param/input/gen_cv.sh" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/param/input/gen_cv.sh"; fi
if [ -e "mdin/qmmm/semiempirical/param/input/cv.rst" ]; then git rm -f "mdin/qmmm/semiempirical/param/input/cv.rst" 2>/dev/null || rm -f "mdin/qmmm/semiempirical/param/input/cv.rst"; fi
if [ -e "ambertools/pytraj/rmsd-single.py" ]; then git rm -f "ambertools/pytraj/rmsd-single.py" 2>/dev/null || rm -f "ambertools/pytraj/rmsd-single.py"; fi
if [ -e "mlp/torchmdnet/train2.sh" ]; then git rm -f "mlp/torchmdnet/train2.sh" 2>/dev/null || rm -f "mlp/torchmdnet/train2.sh"; fi
echo 'Done.'