import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_original import ORIGINAL, SCRIPTS, canonical_source_digest, patch_source

ROOT = Path(__file__).resolve().parents[1]


class TestPortability(unittest.TestCase):
    def test_every_original_path_is_patched_and_compiles(self):
        for name in SCRIPTS.values():
            code = patch_source((ORIGINAL / name).read_text(encoding="utf-8"), name)
            self.assertNotIn("/mnt" + "/data", code)
            compile(code, name, "exec")

    def test_canonical_hash_is_line_ending_independent(self):
        source = "x = 1\ny = 2\n"
        self.assertEqual(canonical_source_digest(source), canonical_source_digest(source.replace("\n", "\r\n")))

    def test_child_scripts_start_from_scripts_directory(self):
        scripts = ROOT / "scripts"
        for command in (
            ["prepare_active_contract.py", "--help"],
            ["run_original.py", "--help"],
            ["postprocess_body_bands.py", "--help"],
        ):
            result = subprocess.run(
                [sys.executable, *command], cwd=scripts,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_works_from_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="FVG Test User (3) ") as directory:
            result = subprocess.run(
                [sys.executable, "-c", "import fvg_research; print(fvg_research.__file__)"],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
