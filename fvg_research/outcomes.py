from __future__ import annotations
import numpy as np
import pandas as pd

def first_touch_time(df: pd.DataFrame, row, max_bars: int = 10_000):
    loc = df.index.searchsorted(row.name)
    fut = df.iloc[loc+1:loc+1+max_bars]
    if row.direction == 1:
        hit = fut["low"].to_numpy() <= row.near
    else:
        hit = fut["high"].to_numpy() >= row.near
    idx = np.flatnonzero(hit)
    return None if not len(idx) else fut.index[idx[0]]

def barrier_race(df: pd.DataFrame, start_ts, direction: int, near: float, far: float,
                 reject_distance: float | None = None, max_bars: int = 10_000):
    """Return 1 if rejection wins, 0 if far edge wins, NaN if unresolved/ambiguous."""
    loc = df.index.searchsorted(start_ts)
    fut = df.iloc[loc+1:loc+1+max_bars]
    width = abs(near - far)
    reject = near + direction * (reject_distance if reject_distance is not None else width)
    for _, b in fut.iterrows():
        if direction == 1:
            win = b.high >= reject
            lose = b.low <= far
        else:
            win = b.low <= reject
            lose = b.high >= far
        if win and lose:
            return np.nan
        if win:
            return 1.0
        if lose:
            return 0.0
    return np.nan

def midpoint_race(df: pd.DataFrame, touch_ts, direction: int, near: float, far: float,
                  max_bars: int = 10_000):
    """From the bar after CE touch, near edge wins=1; far edge wins=0."""
    loc = df.index.searchsorted(touch_ts)
    fut = df.iloc[loc+1:loc+1+max_bars]
    for _, b in fut.iterrows():
        if direction == 1:
            win = b.high >= near
            lose = b.low <= far
        else:
            win = b.low <= near
            lose = b.high >= far
        if win and lose:
            return np.nan
        if win:
            return 1.0
        if lose:
            return 0.0
    return np.nan

def close_depth(direction: int, lower: float, upper: float, close: float) -> float:
    width = upper - lower
    if width <= 0:
        return np.nan
    return (upper - close) / width if direction == 1 else (close - lower) / width
