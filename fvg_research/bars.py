from __future__ import annotations

import numpy as np
import pandas as pd

TF_MAP = {
    "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
    "10m": "10min", "15m": "15min", "30m": "30min",
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h",
    "8H": "8h", "12H": "12h", "1D": "1D",
}


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf not in TF_MAP:
        raise ValueError(f"Unsupported timeframe {tf!r}; choose one of {sorted(TF_MAP)}")
    if tf == "1m":
        return df.copy()
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("resample_ohlcv requires a timezone-aware DatetimeIndex")
    local = df.tz_convert("America/New_York")
    aggregations = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for column in ("symbol", "trade_date"):
        if column in local.columns:
            aggregations[column] = "last"
    out = local.resample(
        TF_MAP[tf], label="right", closed="right", origin="start_day", offset="18h"
    ).agg(aggregations)
    return out.dropna(subset=["open", "high", "low", "close"]).tz_convert("UTC")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    previous = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(n, min_periods=n).mean()


def market_state(df: pd.DataFrame) -> pd.DataFrame:
    """Causal state features: thresholds never use future observations."""
    x = df.copy()
    x["atr14"] = atr(x, 14)
    x["ret20"] = x["close"].pct_change(20)
    x["volatility"] = x["atr14"] / x["close"]
    x["trend"] = np.sign(x["ret20"]).fillna(0).astype(int)
    history = x["volatility"].shift(1).rolling(10_000, min_periods=100)
    low = history.quantile(0.33)
    high = history.quantile(0.67)
    regime = np.select(
        [x["volatility"] < low, x["volatility"] > high],
        ["low", "high"], default="normal",
    )
    x["vol_regime"] = pd.Series(regime, index=x.index, dtype="string")
    eastern = x.index.tz_convert("America/New_York")
    minute_of_day = eastern.hour * 60 + eastern.minute
    x["session"] = np.select(
        [
            (minute_of_day >= 18 * 60) | (minute_of_day < 2 * 60),
            (minute_of_day >= 2 * 60) & (minute_of_day < 8 * 60),
            (minute_of_day >= 8 * 60) & (minute_of_day < 9 * 60 + 30),
            (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 12 * 60),
            (minute_of_day >= 12 * 60) & (minute_of_day < 13 * 60 + 30),
            (minute_of_day >= 13 * 60 + 30) & (minute_of_day < 16 * 60),
        ],
        ["asia", "london", "ny_premarket", "ny_am", "ny_lunch", "ny_pm"],
        default="postmarket",
    )
    x["tod_30m"] = (minute_of_day // 30).astype(int)
    return x
