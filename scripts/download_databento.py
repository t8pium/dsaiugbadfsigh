from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import databento as db
from fvg_research.config import RAW, START, END

key = os.environ.get("DATABENTO_API_KEY")
if not key:
    raise SystemExit("Set DATABENTO_API_KEY before running this script.")

client = db.Historical(key)
store = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    schema="ohlcv-1m",
    stype_in="parent",
    symbols="MNQ.FUT",
    start=START,
    end=END,
)
df = store.to_df()
out = RAW / "mnq_ohlcv_1m.parquet"
df.reset_index().to_parquet(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
