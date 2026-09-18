from __future__ import annotations
import pandas as pd
from pathlib import Path
from fvg_research.config import ACTIVE_1M, RESULTS, TICK_SIZE
from fvg_research.io import read_any
from fvg_research.bars import resample_ohlcv, market_state
from fvg_research.fvg import detect_fvgs

def load_tf(tf: str):
    base = read_any(ACTIVE_1M)
    bars = resample_ohlcv(base, tf)
    state = market_state(bars)
    events = detect_fvgs(bars, TICK_SIZE).join(
        state[["session","tod_30m","vol_regime","trend","volatility"]],
        how="left",
        rsuffix="_state",
    )
    if "roll_date" in base.columns and tf == "1m":
        bad = base["roll_date"].reindex(events.index).fillna(False)
        events = events.loc[~bad]
    return bars, state, events

def save(frame: pd.DataFrame, name: str):
    path = RESULTS / name
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")
