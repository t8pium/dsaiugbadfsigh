from __future__ import annotations
import pandas as pd
from fvg_research.config import RAW, PROCESSED
from fvg_research.io import read_any

def cme_trade_date(index: pd.DatetimeIndex) -> pd.Series:
    ct = index.tz_convert("America/Chicago")
    return pd.Series((ct - pd.Timedelta(hours=17)).date, index=index, name="trade_date")

src = RAW / "mnq_ohlcv_1m.parquet"
df = read_any(src)
symbol_col = "symbol" if "symbol" in df.columns else ("raw_symbol" if "raw_symbol" in df.columns else None)
if symbol_col is None:
    raise SystemExit("Input needs symbol or raw_symbol so listed expiries can be separated.")
if symbol_col != "symbol":
    df = df.rename(columns={symbol_col: "symbol"})

df["trade_date"] = cme_trade_date(df.index)
daily = df.groupby(["trade_date","symbol"], observed=True)["volume"].sum()
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
print(f"Wrote {len(active):,} active-contract bars")
print(pq)
print(pkl)
