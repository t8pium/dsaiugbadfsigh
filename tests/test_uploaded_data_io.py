import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from fvg_research.io import read_any, read_many


def sample_frame(start: str, symbol: str = "MNQH6"):
    idx = pd.date_range(start, periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "ts_event": idx,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10, 11, 12],
            "symbol": [symbol] * 3,
        }
    )


class TestUploadedDataIO(unittest.TestCase):
    def test_csv_upload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mnq.csv"
            sample_frame("2026-01-01").to_csv(path, index=False)
            out = read_any(path)
            self.assertEqual(len(out), 3)
            self.assertIn("symbol", out.columns)
            self.assertTrue(isinstance(out.index, pd.DatetimeIndex))

    def test_zip_upload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "inside.csv"
            zip_path = root / "upload.zip"
            sample_frame("2026-01-01").to_csv(csv_path, index=False)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(csv_path, arcname="nested/inside.csv")

            out = read_any(zip_path)
            self.assertEqual(len(out), 3)
            self.assertEqual(out["symbol"].iloc[0], "MNQH6")

    def test_multiple_uploaded_files_are_combined(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p1 = root / "part1.csv"
            p2 = root / "part2.csv"
            sample_frame("2026-01-01").to_csv(p1, index=False)
            sample_frame("2026-01-02", symbol="MNQM6").to_csv(p2, index=False)

            out = read_many([p1, p2])
            self.assertEqual(len(out), 6)
            self.assertEqual(set(out["symbol"]), {"MNQH6", "MNQM6"})


if __name__ == "__main__":
    unittest.main()
