from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import pandas as pd

REQUIRED = {"open", "high", "low", "close", "volume"}
SUPPORTED_SUFFIXES = (
    ".parquet", ".pq", ".csv", ".csv.gz", ".gz",
    ".dbn", ".dbn.zst", ".zst", ".zip",
)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Databento DBNStore.to_df() normally places ts_event on the index.
    if "ts_event" in x.columns:
        x["ts_event"] = pd.to_datetime(x["ts_event"], utc=True)
        x = x.set_index("ts_event")
    elif not isinstance(x.index, pd.DatetimeIndex):
        for c in ("timestamp", "datetime", "time"):
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], utc=True)
                x = x.set_index(c)
                break

    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex or ts_event/timestamp column is required.")

    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")

    missing = REQUIRED.difference(x.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    x = x.sort_index()
    return x


def _is_dbn(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".dbn") or name.endswith(".dbn.zst")


def _read_dbn(path: Path) -> pd.DataFrame:
    import databento as db

    store = db.DBNStore.from_file(path)
    df = store.to_df()
    return normalize_ohlcv(df)


def _read_zip(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(path, "r") as zf, tempfile.TemporaryDirectory(prefix="fvg_upload_") as td:
        root = Path(td).resolve()

        for info in zf.infolist():
            if info.is_dir():
                continue

            # Prevent path traversal when extracting user-provided archives.
            target = (root / Path(info.filename).name).resolve()
            if root not in target.parents:
                continue

            lname = info.filename.lower()
            supported = (
                lname.endswith(".parquet")
                or lname.endswith(".pq")
                or lname.endswith(".csv")
                or lname.endswith(".csv.gz")
                or lname.endswith(".dbn")
                or lname.endswith(".dbn.zst")
            )
            if not supported:
                continue

            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024 * 8)
                    if not chunk:
                        break
                    dst.write(chunk)

            frames.append(read_any(target))

    if not frames:
        raise ValueError(
            "ZIP contains no supported Databento/OHLCV files. "
            "Supported members: .dbn, .dbn.zst, .parquet, .pq, .csv, .csv.gz"
        )

    return normalize_ohlcv(pd.concat(frames, axis=0, sort=False))


def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    name = p.name.lower()

    if name.endswith(".zip"):
        return _read_zip(p)

    if _is_dbn(p):
        return _read_dbn(p)

    if name.endswith(".parquet") or name.endswith(".pq"):
        return normalize_ohlcv(pd.read_parquet(p))

    if name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".gz"):
        return normalize_ohlcv(pd.read_csv(p))

    # A bare .zst upload is sometimes a DBN file renamed by a browser/download.
    if name.endswith(".zst"):
        try:
            return _read_dbn(p)
        except Exception as exc:
            raise ValueError(
                f"Could not read {p.name} as Databento DBN/Zstd: {exc}"
            ) from exc

    raise ValueError(
        f"Unsupported input: {p.name}. "
        "Use .dbn, .dbn.zst, .parquet, .pq, .csv, .csv.gz, or .zip."
    )


def read_many(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No input files supplied.")

    frames = [read_any(p) for p in paths]
    if len(frames) == 1:
        return frames[0]
    return normalize_ohlcv(pd.concat(frames, axis=0, sort=False))
