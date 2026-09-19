import gzip
import io
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fvg_research.io import read_any, read_many


def sample_frame(start: str, symbol: str = "MNQH26"):
    timestamps = pd.date_range(start, periods=3, freq="1min", tz="UTC")
    return pd.DataFrame({
        "ts_event": timestamps, "open": [100., 101., 102.], "high": [101., 102., 103.],
        "low": [99., 100., 101.], "close": [100.5, 101.5, 102.5],
        "volume": [10, 11, 12], "symbol": [symbol] * 3,
    })


class TestUploadedDataIO(unittest.TestCase):
    def test_csv_csv_gz_and_csv_zst(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("bars.csv", "bars.csv.gz", "bars.csv.zst"):
                path = root / name
                sample_frame("2026-01-01").to_csv(path, index=False)
                self.assertEqual(len(read_any(path)), 3)

    def test_parquet_and_pq(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("bars.parquet", "bars.pq"):
                path = Path(directory) / name
                sample_frame("2026-01-01").to_parquet(path, index=False)
                self.assertEqual(len(read_any(path)), 3)

    def test_zip_multiple_duplicate_names_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "batch.zip"
            first = sample_frame("2026-01-01").to_csv(index=False).encode()
            second = sample_frame("2026-01-02", "MNQM26").to_csv(index=False).encode()
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("one/same.csv", first)
                zipped.writestr("two/same.csv", second)
                zipped.writestr("../../escape.csv", first)
            output = read_any(archive)
            self.assertEqual(len(output), 9)
            self.assertFalse((root.parent / "escape.csv").exists())

    def test_malformed_zip_and_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.zip"
            bad.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "Malformed ZIP"):
                read_any(bad)
            text = Path(directory) / "x.txt"
            text.write_text("x")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_any(text)

    def test_missing_ohlc_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.csv"
            sample_frame("2026-01-01").drop(columns="low").to_csv(missing, index=False)
            with self.assertRaisesRegex(ValueError, "Missing required"):
                read_any(missing)
            invalid = Path(directory) / "invalid.csv"
            frame = sample_frame("2026-01-01")
            frame.loc[0, "high"] = 0
            frame.to_csv(invalid, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid OHLC"):
                read_any(invalid)

    def test_multiple_files_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p1, p2 = root / "a.csv", root / "b.csv"
            sample_frame("2026-01-01").to_csv(p1, index=False)
            sample_frame("2026-01-02", "MNQM26").to_csv(p2, index=False)
            self.assertEqual(len(read_many([p1, p2])), 6)
            self.assertEqual(len(read_any(root)), 6)

    def test_dbn_requests_symbol_mapping_and_chunks(self):
        calls = {}
        frame = sample_frame("2026-01-01").set_index("ts_event")

        class Store:
            @classmethod
            def from_file(cls, path):
                calls["path"] = Path(path)
                return cls()
            def to_df(self, **kwargs):
                calls["kwargs"] = kwargs
                return iter([frame.iloc[:2], frame.iloc[2:]])

        fake = types.SimpleNamespace(DBNStore=Store)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"databento": fake}):
            path = Path(directory) / "bars.dbn.zst"
            path.write_bytes(b"fake")
            output = read_any(path)
        self.assertEqual(len(output), 3)
        self.assertEqual(calls["kwargs"], {"map_symbols": True, "count": 500_000})

    def test_zip_associates_each_dbn_sidecar(self):
        inserted = {}
        frame = sample_frame("2026-01-01").set_index("ts_event")

        class Store:
            def __init__(self, path): self.path = Path(path)
            @classmethod
            def from_file(cls, path): return cls(path)
            def insert_symbology_json(self, value):
                inserted.setdefault(self.path.name, []).append(str(value))
            def to_df(self, **kwargs): return frame

        fake = types.SimpleNamespace(DBNStore=Store)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"databento": fake}):
            archive = Path(directory) / "batch.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("a.dbn", b"a")
                zipped.writestr("a.symbology.json", "{}")
                zipped.writestr("b.dbn", b"b")
                zipped.writestr("b.symbology.json", "{}")
            self.assertEqual(len(read_any(archive)), 6)
        self.assertEqual(len(inserted), 2)
        self.assertTrue(all(len(values) == 1 for values in inserted.values()))


if __name__ == "__main__":
    unittest.main()
