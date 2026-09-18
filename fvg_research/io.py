from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"open", "high", "low", "close", "volume"}

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
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

def read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return normalize_ohlcv(pd.read_parquet(p))
    if p.suffix.lower() in {".csv", ".gz"}:
        return normalize_ohlcv(pd.read_csv(p))
    raise ValueError(f"Unsupported input: {p}")
