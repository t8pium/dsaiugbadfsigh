import json
import tempfile
import unittest
from pathlib import Path

from fvg_research.dashboard_helpers import (
    discover_results,
    load_latest_success,
    parse_local_paths,
    safe_upload_name,
)


class TestDashboardHelpers(unittest.TestCase):
    def test_upload_name_is_windows_safe(self):
        self.assertEqual(safe_upload_name(r"..\..\evil.dbn.zst"), "evil.dbn.zst")
        self.assertNotIn(":", safe_upload_name(r"C:\temp\bars.csv"))
        self.assertTrue(safe_upload_name("CON.csv").startswith("databento_upload_"))

    def test_local_paths_preserve_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory(prefix="Test User ") as directory:
            paths = parse_local_paths(f'"{directory}/file one.csv"\n{directory}/folder two')
            self.assertEqual(len(paths), 2)
            self.assertIn("file one.csv", str(paths[0]))

    def test_result_discovery_and_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, runs = root / "results", root / "results" / "_runs"
            runs.mkdir(parents=True)
            data = root / "active.pkl"
            data.write_bytes(b"data")
            output = results / "x.csv"
            output.write_text("a\n1\n")
            (results / "ignored.pkl").write_bytes(b"x")
            payload = {
                "status": "success", "data_mtime_ns": data.stat().st_mtime_ns,
                "generated_files": ["results/x.csv"],
            }
            (runs / "latest_demo.json").write_text(json.dumps(payload))
            self.assertIn(output, discover_results(results))
            self.assertNotIn(results / "ignored.pkl", discover_results(results))
            self.assertIsNotNone(load_latest_success(results, data, "demo"))
            payload["data_mtime_ns"] -= 1
            (runs / "latest_demo.json").write_text(json.dumps(payload))
            self.assertIsNone(load_latest_success(results, data, "demo"))


if __name__ == "__main__":
    unittest.main()
