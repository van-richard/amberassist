"""Worker regressions using the legacy QMHub interface, without scientific jobs."""

import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "wtp" / "wtp_energy.py"
LEGACY = ROOT / "wtp" / "legacy" / "wtp_energy.py"
SETUP_CALLS = ["construct", "setup_simulation", "load_system", "build_model", "add_engine"]


class FakeSimulation:
    def __init__(self, owner):
        self.owner = owner
        self.gradient_reads = 0
        self.full_gradient = np.array(
            [
                [-1.25, 2.5, -3.75, 4.0],
                [5.0, -6.25, 7.5, -8.75],
                [9.0, 10.25, -11.5, -12.75],
            ],
            dtype=np.float64,
        )

    def check_ready(self):
        if self.owner.calls[: len(SETUP_CALLS)] != SETUP_CALLS:
            raise AssertionError("Evaluation occurred before the legacy setup sequence completed")

    @property
    def energy(self):
        self.check_ready()
        self.owner.calls.append("energy")
        # QMHub's legacy scalar need not be a Python float or a 0-D array.
        return np.array([[-123.456789012345]], dtype=np.float64)

    @property
    def energy_gradient(self):
        self.check_ready()
        self.gradient_reads += 1
        if not self.owner.allow_gradient:
            raise AssertionError("Unrequested gradient was evaluated")
        self.owner.calls.append("energy_gradient")
        return self.full_gradient


class FakeQMMM:
    """Expose only the API used by the unchanged legacy worker."""

    instances = []
    allow_gradient = True

    def __init__(self, *, mode, driver, cwd):
        self.mode = mode
        self.driver = driver
        self.io = types.SimpleNamespace(cwd=Path(cwd))
        # An unsorted, noncontiguous selection catches accidental full-system
        # output, reordering, or selecting only the first n columns.
        self.system = types.SimpleNamespace(qm_index=np.array([3, 1]))
        self.calls = ["construct"]
        self.__class__.instances.append(self)

    def setup_simulation(self, protocol="md"):
        if not self.io.cwd.is_dir():
            raise AssertionError("The working directory must exist before setup")
        self.protocol = protocol
        self.simulation = FakeSimulation(self)
        self.calls.append("setup_simulation")

    def load_system(self, fin):
        self.loaded = Path(fin)
        self.calls.append("load_system")

    def build_model(self, **options):
        self.model_options = options
        self.calls.append("build_model")

    def add_engine(self, engine, options):
        self.engine = engine
        self.engine_options = options.copy()
        self.calls.append("add_engine")


class WtpEnergyWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wtp-worker-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.input_dir = self.base / "inputs"
        self.input_dir.mkdir()
        self.fin = self.input_dir / "qmmm.inp_0007"
        self.fin.write_text("Mock exchange input; no scientific parsing is performed.\n")
        self.outdir = self.base / "out"

        FakeQMMM.instances = []
        FakeQMMM.allow_gradient = True
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        for variable in (
            "QCHEM_MEM", "QCHEM_SCF_CONV", "QCHEM_MAX_SCF", "QCHEM_DFT_GRID",
            "WTP_METHOD", "WTP_BASIS",
        ):
            os.environ.pop(variable, None)

        self.fake_qmhub = types.ModuleType("qmhub")
        self.fake_qmhub.__path__ = []
        self.fake_qmhub.QMMM = FakeQMMM
        with mock.patch.dict(sys.modules, {"qmhub": self.fake_qmhub}):
            spec = importlib.util.spec_from_file_location("wtp_worker_under_test", WORKER)
            self.worker = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(self.worker)

    def run_worker(self, want_grad=False):
        return self.worker.run_wtp_qchem(
            self.fin, "b3lyp", "6-31+gd", want_grad,
            cwd=self.outdir / "work" / self.fin.name,
        )

    def invoke_main(self, fin=None, outdir=None, grad=False, method="b3lyp"):
        arguments = [
            "wtp_energy.py", "--inp", str(fin or self.fin),
            "--outdir", str(outdir or self.outdir),
            "--method", method, "--basis", "6-31+gd",
        ]
        if grad:
            arguments.append("--grad")
        with mock.patch.object(sys, "argv", arguments):
            self.worker.main()

    def test_supported_legacy_api_order_and_default_settings(self):
        self.run_worker()
        instance = FakeQMMM.instances[-1]
        self.assertEqual(instance.calls, SETUP_CALLS + ["energy"])
        self.assertEqual(instance.mode, "text")
        self.assertEqual(instance.driver, "sander")
        self.assertEqual(instance.protocol, "md")
        self.assertEqual(instance.loaded, self.fin)
        self.assertEqual(
            instance.model_options,
            {"switching_type": "lrec", "cutoff": 10.0, "swdist": None, "pbc": True},
        )
        self.assertEqual(instance.engine, "qchem")
        self.assertEqual(
            instance.engine_options,
            {"method": "b3lyp", "basis": "6-31+gd", "scf_convergence": "9"},
        )
        self.assertTrue(instance.io.cwd.is_dir())

    def test_qchem_environment_overrides_are_opt_in(self):
        with mock.patch.dict(os.environ, {
            "QCHEM_MEM": "16384", "QCHEM_SCF_CONV": "11",
            "QCHEM_MAX_SCF": "250", "QCHEM_DFT_GRID": "3",
        }):
            self.run_worker()
        self.assertEqual(
            FakeQMMM.instances[-1].engine_options,
            {
                "method": "b3lyp", "basis": "6-31+gd", "scf_convergence": "11",
                "mem_total": "16384", "max_scf_cycles": "250", "dft_grid": "3",
            },
        )

    def test_one_element_energy_becomes_scalar_with_explicit_reference_placeholder(self):
        result = self.run_worker()
        self.assertIsInstance(result["E_QM"], float)
        self.assertEqual(result["E_QM"], -123.456789012345)
        self.assertEqual(result["E_ref"], 0.0)
        self.assertEqual(result["dE"], result["E_QM"])
        self.assertEqual(result["meta"], {
            "inp": self.fin.name,
            "engine": "qchem",
            "method": "b3lyp",
            "basis": "6-31+gd",
            "reference_energy_source": "zero_placeholder",
        })

    def test_requested_gradient_preserves_positive_sign_qm_order_axes_and_dtype(self):
        result = self.run_worker(want_grad=True)
        instance = FakeQMMM.instances[-1]
        expected = instance.simulation.full_gradient[:, instance.system.qm_index]
        np.testing.assert_array_equal(result["forces"], expected.astype(np.float32))
        self.assertEqual(result["forces"].shape, (3, 2))
        self.assertEqual(result["forces"].dtype, np.dtype("float32"))
        self.assertEqual(instance.calls, SETUP_CALLS + ["energy", "energy_gradient"])
        self.assertEqual(result["meta"]["forces_quantity"], "positive_qm_subset_energy_gradient")
        self.assertEqual(result["meta"]["forces_axes"], ["xyz", "qm_atom"])

    def test_gradient_is_neither_evaluated_nor_returned_without_grad(self):
        FakeQMMM.allow_gradient = False
        result = self.run_worker()
        self.assertNotIn("forces", result)
        self.assertEqual(FakeQMMM.instances[-1].simulation.gradient_reads, 0)

    def test_cli_npz_contains_scalar_energies_gradients_and_json_metadata(self):
        self.invoke_main(grad=True)
        with np.load(self.outdir / "wtp_frame_0007.npz", allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {"E_QM", "E_ref", "dE", "forces", "meta"})
            self.assertEqual(saved["E_QM"].shape, ())
            self.assertEqual(saved["E_QM"].item(), -123.456789012345)
            self.assertEqual(saved["E_ref"].item(), 0.0)
            self.assertEqual(saved["dE"].item(), saved["E_QM"].item())
            expected = FakeQMMM.instances[-1].simulation.full_gradient[:, [3, 1]]
            np.testing.assert_array_equal(saved["forces"], expected.astype(np.float32))
            self.assertEqual(saved["forces"].dtype, np.dtype("float32"))
            metadata = json.loads(saved["meta"].item())
        self.assertEqual(metadata, {
            "inp": self.fin.name,
            "engine": "qchem",
            "method": "b3lyp",
            "basis": "6-31+gd",
            "reference_energy_source": "zero_placeholder",
            "forces_quantity": "positive_qm_subset_energy_gradient",
            "forces_axes": ["xyz", "qm_atom"],
        })

    def test_cli_omits_gradients_and_their_metadata_without_grad(self):
        FakeQMMM.allow_gradient = False
        self.invoke_main()
        with np.load(self.outdir / "wtp_frame_0007.npz", allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {"E_QM", "E_ref", "dE", "meta"})
            metadata = json.loads(saved["meta"].item())
        self.assertNotIn("forces_quantity", metadata)
        self.assertNotIn("forces_axes", metadata)
        self.assertEqual(FakeQMMM.instances[-1].simulation.gradient_reads, 0)

    def test_frames_and_theory_output_directories_isolate_qchem_work(self):
        other = self.input_dir / "qmmm.inp_0008"
        other.write_text(self.fin.read_text())
        second_theory = self.base / "second-theory"
        self.invoke_main()
        self.invoke_main(fin=other)
        self.invoke_main(outdir=second_theory, method="pbe0")
        directories = [instance.io.cwd.resolve() for instance in FakeQMMM.instances]
        self.assertEqual(directories, [
            (self.outdir / "work" / self.fin.name).resolve(),
            (self.outdir / "work" / other.name).resolve(),
            (second_theory / "work" / self.fin.name).resolve(),
        ])
        self.assertEqual(len(set(directories)), 3)
        self.assertTrue(all(directory.is_dir() for directory in directories))

    def test_missing_input_fails_before_backend_or_output_creation(self):
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as error:
                self.invoke_main(fin=self.input_dir / "qmmm.inp_9999")
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Input file does not exist", stderr.getvalue())
        self.assertEqual(FakeQMMM.instances, [])
        self.assertFalse(self.outdir.exists())

    def test_suffix_without_digits_fails_before_backend_or_output_creation(self):
        invalid = self.input_dir / "qmmm42.inp_invalid"
        invalid.write_text(self.fin.read_text())
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as error:
                self.invoke_main(fin=invalid)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Cannot parse frame index", stderr.getvalue())
        self.assertEqual(FakeQMMM.instances, [])
        self.assertFalse(self.outdir.exists())

    def test_fake_legacy_outputs_match_worker_energy_and_qm_gradients(self):
        reference_dir = self.base / "legacy-output"
        reference_dir.mkdir()
        units = types.ModuleType("qmhub.units")
        units.CODATA08_BOHR_TO_A = 0.529177210903  # Imported but unused by the reference.
        previous = Path.cwd()
        try:
            os.chdir(reference_dir)
            with mock.patch.dict(sys.modules, {"qmhub": self.fake_qmhub, "qmhub.units": units}):
                with mock.patch.object(sys, "argv", ["wtp_energy.py", str(self.fin)]):
                    runpy.run_path(str(LEGACY), run_name="__main__")
        finally:
            os.chdir(previous)
        legacy_instance = FakeQMMM.instances[-1]
        energy = np.load(reference_dir / "qmmm_b3lyp_6-31+gd_energy.npy", allow_pickle=False)
        gradient = np.load(reference_dir / "qmmm_b3lyp_6-31+gd_forces.npy", allow_pickle=False)
        result = self.run_worker(want_grad=True)
        self.assertEqual(result["E_QM"], energy.item())
        np.testing.assert_array_equal(result["forces"], gradient[0].astype(np.float32))
        instance = FakeQMMM.instances[-1]
        self.assertEqual(instance.engine_options, legacy_instance.engine_options)
        self.assertEqual(instance.model_options, legacy_instance.model_options)


if __name__ == "__main__":
    unittest.main()
