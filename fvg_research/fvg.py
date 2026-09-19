from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import atr


def detect_fvgs(df: pd.DataFrame, tick_size: float = 0.25) -> pd.DataFrame:
    """Detect three-candle FVGs at candle C close without lookahead."""
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    a_high = df["high"].shift(2)
    a_low = df["low"].shift(2)
    bull_width = df["low"] - a_high
    bear_width = a_low - df["high"]
    epsilon = tick_size * 1e-9
    bull = bull_width >= tick_size - epsilon
    bear = bear_width >= tick_size - epsilon
    direction = np.where(bull, 1, np.where(bear, -1, 0))
    near = pd.Series(np.where(bull, df["low"], np.where(bear, df["high"], np.nan)), index=df.index)
    far = pd.Series(np.where(bull, a_high, np.where(bear, a_low, np.nan)), index=df.index)
    lower = pd.concat([near, far], axis=1).min(axis=1)
    upper = pd.concat([near, far], axis=1).max(axis=1)

    events = pd.DataFrame(index=df.index)
    events["direction"] = direction
    events["near"] = near
    events["far"] = far
    events["lower"] = lower
    events["upper"] = upper
    events["mid"] = (lower + upper) / 2
    events["width"] = upper - lower
    events["ticks"] = events["width"] / tick_size
    events["close_at_formation"] = df["close"]
    events["atr14"] = atr(df, 14)
    events["width_atr"] = events["width"] / events["atr14"]
    events["distance"] = np.where(
        events["direction"] == 1,
        (df["close"] - events["upper"]).clip(lower=0),
        (events["lower"] - df["close"]).clip(lower=0),
    )
    events["distance_atr"] = events["distance"] / events["atr14"]
    events["body_b"] = (df["close"].shift(1) - df["open"].shift(1)).abs()
    events["body_b_atr"] = events["body_b"] / events["atr14"]
    return events.loc[events["direction"] != 0].dropna(subset=["lower", "upper", "atr14"])


def forward_extrema(df: pd.DataFrame, horizon: int):
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    high = df["high"].shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    low = df["low"].shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    return high, low


def _at_horizon(df, zones, horizon, level, time_col=None):
    high, low = forward_extrema(df, horizon)
    index = pd.DatetimeIndex(zones[time_col]) if time_col else zones.index
    maximum = high.reindex(index).to_numpy()
    minimum = low.reindex(index).to_numpy()
    direction = zones["direction"].to_numpy()
    threshold = zones[level].to_numpy()
    hit = np.where(direction == 1, minimum <= threshold, maximum >= threshold)
    return pd.Series(hit, index=zones.index)


def touch_at_horizon(df, zones, horizon, time_col=None):
    return _at_horizon(df, zones, horizon, "near", time_col)


def full_at_horizon(df, zones, horizon, time_col=None):
    return _at_horizon(df, zones, horizon, "far", time_col)


def midpoint_at_horizon(df, zones, horizon, time_col=None):
    return _at_horizon(df, zones, horizon, "mid", time_col)
