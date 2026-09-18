from __future__ import annotations
import numpy as np
import pandas as pd
from .bars import atr

def detect_fvgs(df: pd.DataFrame, tick_size: float = 0.25) -> pd.DataFrame:
    """Detect FVGs causally. Each event exists only after candle C closes."""
    a_high = df["high"].shift(2)
    a_low = df["low"].shift(2)
    bull = df["low"] > a_high
    bear = df["high"] < a_low

    direction = np.where(bull, 1, np.where(bear, -1, 0))
    near = np.where(bull, df["low"], np.where(bear, df["high"], np.nan))
    far = np.where(bull, a_high, np.where(bear, a_low, np.nan))
    lower = np.minimum(near, far)
    upper = np.maximum(near, far)

    e = pd.DataFrame(index=df.index)
    e["direction"] = direction
    e["near"] = near
    e["far"] = far
    e["lower"] = lower
    e["upper"] = upper
    e["mid"] = (lower + upper) / 2
    e["width"] = upper - lower
    e["ticks"] = e["width"] / tick_size
    e["close_at_formation"] = df["close"]
    e["atr14"] = atr(df, 14)
    e["width_atr"] = e["width"] / e["atr14"]
    e["distance"] = np.where(
        e["direction"] == 1,
        (df["close"] - e["upper"]).clip(lower=0),
        (e["lower"] - df["close"]).clip(lower=0),
    )
    e["distance_atr"] = e["distance"] / e["atr14"]
    e["body_b"] = (df["close"].shift(1) - df["open"].shift(1)).abs()
    e["body_b_atr"] = e["body_b"] / e["atr14"]
    return e[e["direction"] != 0].dropna(subset=["lower","upper","atr14"])

def forward_extrema(df: pd.DataFrame, horizon: int):
    """Future extrema over bars t+1..t+horizon, excluding the event bar."""
    hi = df["high"].shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    lo = df["low"].shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    return hi, lo

def touch_at_horizon(df: pd.DataFrame, zones: pd.DataFrame, horizon: int, time_col: str | None = None) -> pd.Series:
    hi, lo = forward_extrema(df, horizon)
    idx = pd.DatetimeIndex(zones[time_col]) if time_col else zones.index
    mx = hi.reindex(idx).to_numpy()
    mn = lo.reindex(idx).to_numpy()
    d = zones["direction"].to_numpy()
    near = zones["near"].to_numpy()
    hit = np.where(d == 1, mn <= near, mx >= near)
    return pd.Series(hit, index=zones.index)

def full_at_horizon(df: pd.DataFrame, zones: pd.DataFrame, horizon: int, time_col: str | None = None) -> pd.Series:
    hi, lo = forward_extrema(df, horizon)
    idx = pd.DatetimeIndex(zones[time_col]) if time_col else zones.index
    mx = hi.reindex(idx).to_numpy()
    mn = lo.reindex(idx).to_numpy()
    d = zones["direction"].to_numpy()
    far = zones["far"].to_numpy()
    hit = np.where(d == 1, mn <= far, mx >= far)
    return pd.Series(hit, index=zones.index)

def midpoint_at_horizon(df: pd.DataFrame, zones: pd.DataFrame, horizon: int, time_col: str | None = None) -> pd.Series:
    hi, lo = forward_extrema(df, horizon)
    idx = pd.DatetimeIndex(zones[time_col]) if time_col else zones.index
    mx = hi.reindex(idx).to_numpy()
    mn = lo.reindex(idx).to_numpy()
    d = zones["direction"].to_numpy()
    mid = zones["mid"].to_numpy()
    hit = np.where(d == 1, mn <= mid, mx >= mid)
    return pd.Series(hit, index=zones.index)
