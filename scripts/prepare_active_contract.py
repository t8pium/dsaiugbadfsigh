from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from fvg_research.config import RAW, PROCESSED
from fvg_research.io import read_many

OUTRIGHT = re.compile(r"^MNQ[HMUZ]\d$")


def cme_trade_date(index: pd.DatetimeIndex) -> pd.Series:
    """Match the published study: ET calendar date, with >=18:00 ET assigned to next date."""
    et = index.tz_convert("America/New_York")
    date = pd.Series(et.date, index=index)
    after_open = et.hour >= 18
    date.loc[after_open] = (et[after_open] + pd.Timedelta(days=1)).date
    return date.rename("trade_date")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build the active MNQ one-minute series from one or more Databento/OHLCV files."
    )
    p.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help=(
            "Input file. Repeat --input for multiple files. "
            "Supports .dbn, .dbn.zst, .parquet, .csv, .csv.gz and .zip."
        ),
    )
    return p.parse_args()


args = parse_args()
inputs = [Path(x) for x in args.inputs] if args.inputs else [RAW / "mnq_ohlcv_1m.parquet"]

missing = [p for p in inputs if not p.exists()]
if missing:
    raise SystemExit("Input file(s) not found: " + ", ".join(map(str, missing)))

print("Reading input:")
for p in inputs:
    print(" -", p)

df = read_many(inputs)

symbol_col = "symbol" if "symbol" in df.columns else ("raw_symbol" if "raw_symbol" in df.columns else None)
if symbol_col is None:
    raise SystemExit(
        "Input needs symbol or raw_symbol so listed expiries can be separated. "
        "If a DBN file has no resolved symbols, export/download it with symbology mappings included."
    )
if symbol_col != "symbol":
    df = df.rename(columns={symbol_col: "symbol"})

# The Databento parent export contains calendar spreads as well as outright contracts.
# The published study kept only quarterly MNQ outrights such as MNQH6/MNQM6/MNQU6/MNQZ6.
df = df[df["symbol"].astype(str).str.match(OUTRIGHT)].copy()
if df.empty:
    raise SystemExit(
        "No quarterly MNQ outright symbols matched ^MNQ[HMUZ]\\d$. "
        "Make sure this is an MNQ parent-symbol OHLCV-1m export."
    )

df["trade_date"] = cme_trade_date(df.index)

# Published construction: highest total daily volume outright contract for each CME trade date.
daily = df.groupby(["trade_date", "symbol"], observed=True)["volume"].sum()
active_map = daily.groupby(level=0).idxmax().map(lambda x: x[1]).rename("active_symbol")
df = df.join(active_map, on="trade_date")
active = df[df["symbol"] == df["active_symbol"]].copy()
active = active[~active.index.duplicated(keep="last")].sort_index()

active = active.reset_index()
if "ts_event" not in active.columns:
    active = active.rename(columns={active.columns[0]: "ts_event"})
active["ts_event"] = pd.to_datetime(active["ts_event"], utc=True)

pq = PROCESSED / "mnq_active_1m.parquet"
pkl = PROCESSED / "active_mnq.pkl"
active.to_parquet(pq, index=False)
active.to_pickle(pkl)

changes = active["symbol"].ne(active["symbol"].shift()).sum()
print(f"Wrote {len(active):,} active-contract bars")
print(f"Contracts: {active['symbol'].nunique()} | contract segments: {changes}")
print(f"Start: {active['ts_event'].min()} | end: {active['ts_event'].max()}")
print(f"Duplicate timestamps: {active['ts_event'].duplicated().sum()}")
print(f"Missing OHLC: {active[['open','high','low','close']].isna().sum().sum()}")
print(pq)
print(pkl)
