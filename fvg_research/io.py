from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

REQUIRED = {"open", "high", "low", "close", "volume"}
DATA_SUFFIXES = (".parquet", ".pq", ".csv", ".csv.gz", ".csv.zst", ".dbn", ".dbn.zst")
MAX_ZIP_MEMBERS = 10_000
MAX_ZIP_EXPANDED = 20 * 1024**3
MAX_COMPRESSION_RATIO = 1_000


def supported_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(DATA_SUFFIXES) or name.endswith(".zip")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize vendor/tabular OHLCV to a sorted, UTC DatetimeIndex."""
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "ts_event" in x.columns:
        x["ts_event"] = pd.to_datetime(x["ts_event"], utc=True, errors="raise")
        x = x.set_index("ts_event")
    elif not isinstance(x.index, pd.DatetimeIndex):
        for column in ("timestamp", "datetime", "time", "date"):
            if column in x.columns:
                x[column] = pd.to_datetime(x[column], utc=True, errors="raise")
                x = x.set_index(column)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex or ts_event/timestamp column is required.")
    x.index = pd.to_datetime(x.index, utc=True)
    x.index.name = "ts_event"
    missing = REQUIRED.difference(x.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    for column in REQUIRED:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    if x[list(REQUIRED)].isna().any().any():
        counts = x[list(REQUIRED)].isna().sum()
        bad = {k: int(v) for k, v in counts.items() if v}
        raise ValueError(f"OHLCV data contains missing/non-numeric values: {bad}")
    if (x["high"] < x[["open", "low", "close"]].max(axis=1)).any() or (
        x["low"] > x[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Invalid OHLC geometry: high/low does not contain open and close.")
    return x.sort_index(kind="stable")


def _is_dbn(path: Path) -> bool:
    return path.name.lower().endswith((".dbn", ".dbn.zst"))


def _sidecar_key(path: Path) -> str:
    name = re.sub(r"^\d{5}_", "", path.name.lower())
    for suffix in (".dbn.zst", ".dbn", ".symbology.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip("._-")
    return path.stem.lower()


def _associated_sidecars(dbn_path: Path, sidecars: Iterable[Path]) -> list[Path]:
    choices = list(sidecars)
    if not choices:
        return []
    key = _sidecar_key(dbn_path)
    exact = [p for p in choices if _sidecar_key(p) == key]
    if exact:
        return exact
    prefixed = [p for p in choices if key.startswith(_sidecar_key(p)) or _sidecar_key(p).startswith(key)]
    if prefixed:
        return prefixed
    return choices if len(choices) == 1 else []


def _insert_symbology(store, sidecar: Path) -> None:
    payload = sidecar.read_text(encoding="utf-8")
    attempts = (str(sidecar), payload, json.loads(payload))
    last: Exception | None = None
    for value in attempts:
        try:
            store.insert_symbology_json(value)
            return
        except (TypeError, ValueError, OSError) as exc:
            last = exc
    raise ValueError(f"Could not apply Databento symbology sidecar {sidecar.name}: {last}")


def _dbn_frames(path: Path, sidecars: Iterable[Path] = ()) -> Iterator[pd.DataFrame]:
    try:
        import databento as db
    except ImportError as exc:
        raise ValueError("Reading DBN requires the pinned 'databento' package.") from exc
    store = db.DBNStore.from_file(path)
    for sidecar in _associated_sidecars(path, sidecars):
        _insert_symbology(store, sidecar)
    try:
        converted = store.to_df(map_symbols=True, count=500_000)
    except TypeError:
        converted = store.to_df(map_symbols=True)
    if isinstance(converted, pd.DataFrame):
        converted = [converted]
    for frame in converted:
        normalized = normalize_ohlcv(frame)
        if "symbol" not in normalized.columns:
            for alias in ("raw_symbol", "stype_out_symbol"):
                if alias in normalized.columns:
                    normalized = normalized.rename(columns={alias: "symbol"})
                    break
        yield normalized


def _tabular_frames(path: Path) -> Iterator[pd.DataFrame]:
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        yield normalize_ohlcv(pd.read_parquet(path))
    elif name.endswith((".csv", ".csv.gz", ".csv.zst")):
        reader = pd.read_csv(path, chunksize=500_000)
        try:
            for chunk in reader:
                yield normalize_ohlcv(chunk)
        finally:
            reader.close()
    else:
        raise ValueError(
            f"Unsupported input: {path.name}. Use .dbn, .dbn.zst, .parquet, "
            ".pq, .csv, .csv.gz, .csv.zst, or .zip."
        )


def _safe_extract_zip(path: Path, root: Path) -> tuple[list[Path], list[Path]]:
    data_files: list[Path] = []
    sidecars: list[Path] = []
    total = 0
    try:
        archive = zipfile.ZipFile(path, "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"Malformed ZIP archive {path.name}: {exc}") from exc
    with archive:
        members = [x for x in archive.infolist() if not x.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError(f"ZIP contains too many members ({len(members):,}).")
        for number, info in enumerate(members, 1):
            if info.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP members are not supported.")
            total += info.file_size
            if total > MAX_ZIP_EXPANDED:
                raise ValueError("ZIP expands beyond the 20 GiB safety limit.")
            if info.file_size and info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise ValueError(f"Suspicious ZIP compression ratio for {info.filename!r}.")
            basename = Path(info.filename.replace("\\", "/")).name
            if not basename:
                continue
            target = root / f"{number:05d}_{basename}"
            lower = basename.lower()
            if not (lower.endswith(DATA_SUFFIXES) or lower.endswith(".json")):
                continue
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024**2)
            if lower.endswith(".json"):
                sidecars.append(target)
            else:
                data_files.append(target)
    if not data_files:
        raise ValueError("ZIP contains no supported market-data files.")
    return data_files, sidecars


def iter_frames(path: str | Path) -> Iterator[pd.DataFrame]:
    """Yield normalized frames without requiring all inputs in memory at once."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_dir():
        children = sorted(p for p in source.iterdir() if p.is_file() and supported_file(p))
        if not children:
            raise ValueError(f"Directory contains no supported files: {source}")
        for child in children:
            yield from iter_frames(child)
        return
    if source.name.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory(prefix="fvg_upload_") as temp:
            data_files, sidecars = _safe_extract_zip(source, Path(temp))
            for item in data_files:
                if _is_dbn(item):
                    yield from _dbn_frames(item, sidecars)
                else:
                    yield from _tabular_frames(item)
        return
    if _is_dbn(source) or source.name.lower().endswith(".zst") and not source.name.lower().endswith(".csv.zst"):
        try:
            yield from _dbn_frames(source)
        except Exception as exc:
            raise ValueError(f"Could not read {source.name} as Databento DBN: {exc}") from exc
        return
    yield from _tabular_frames(source)


def read_any(path: str | Path) -> pd.DataFrame:
    frames = list(iter_frames(path))
    if not frames:
        raise ValueError(f"No data rows found in {path}")
    return normalize_ohlcv(pd.concat(frames, axis=0, sort=False, copy=False))


def read_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    sources = list(paths)
    if not sources:
        raise ValueError("No input files supplied.")
    frames = [frame for source in sources for frame in iter_frames(source)]
    if not frames:
        raise ValueError("No data rows found in supplied files.")
    return normalize_ohlcv(pd.concat(frames, axis=0, sort=False, copy=False))
