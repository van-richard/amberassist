import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "io" / "gen_cvs.py"
SPEC = importlib.util.spec_from_file_location("gen_cvs", SCRIPT_PATH)
gen_cvs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gen_cvs)


class GenCvsTests(unittest.TestCase):
    def test_two_stage_interpolation_preserves_default_window_values(self):
        values = [
            gen_cvs.restraint_value_for_window("PA1", 5.0, 1.0, 4.5, window, 40)
            for window in (0, 19, 20, 39)
        ]
        self.assertEqual(values, [5.0, 1.0, 1.0, 4.5])

    def test_render_restraint_block_preserves_amber_format(self):
        block = gen_cvs.render_restraint_block(
            "PA1",
            ":332&@NZ",
            ":471&@H6",
            10,
            20,
            5.0,
        )
        self.assertEqual(
            block,
            "# PA1 :332&@NZ :471&@H6\n"
            " &rst\n"
            "  iat=10,20,\n"
            "  r1=0, r2=5.00, r3=5.00, r4=10,\n"
            "  rk2=150.0, rk3=150.0,\n"
            " &end\n",
        )

    def test_output_pattern_formats_window(self):
        path = gen_cvs.output_path_for_window("../{window:02d}/cv.rst", 7)
        self.assertEqual(path, Path("../07/cv.rst"))


if __name__ == "__main__":
    unittest.main()
