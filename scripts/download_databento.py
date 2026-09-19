from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import databento as db

from fvg_research.config import END, RAW, START


def yearly_ranges(start: str, end: str):
    first = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    cursor = first
    while cursor < stop:
        boundary = min(date(cursor.year + 1, 1, 1), stop)
        yield cursor.isoformat(), boundary.isoformat()
        cursor = boundary


def main() -> int:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY was not supplied.", file=sys.stderr)
        return 2
    RAW.mkdir(parents=True, exist_ok=True)
    client = db.Historical(key)
    for start, end in yearly_ranges(START, END):
        target = RAW / f"mnq_ohlcv_1m_{start}_{end}.dbn.zst"
        if target.is_file() and target.stat().st_size:
            print(f"Already present: {target.name}", flush=True)
            continue
        partial = target.with_name(target.stem + ".partial.dbn.zst")
        if partial.exists():
            partial.unlink()
        print(f"Downloading {start} through {end} (end exclusive)...", flush=True)
        try:
            client.timeseries.get_range(
                dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="parent",
                symbols="MNQ.FUT", start=start, end=end, path=partial,
            )
        except Exception:
            if partial.exists():
                partial.unlink()
            raise
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Databento did not create {partial}")
        partial.replace(target)
        print(f"Wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
