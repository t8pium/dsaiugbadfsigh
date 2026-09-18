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
    if tf == "1m":
        return df.copy()
    rule = TF_MAP[tf]
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def market_state(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["atr14"] = atr(x, 14)
    x["ret20"] = x["close"].pct_change(20)
    x["volatility"] = x["atr14"] / x["close"]
    x["trend"] = np.sign(x["ret20"]).fillna(0).astype(int)
    q = x["volatility"].rolling(10_000, min_periods=500).quantile
    # global quantiles are intentionally simpler and reproducible.
    lo, hi = x["volatility"].quantile([0.33, 0.67])
    x["vol_regime"] = pd.cut(
        x["volatility"], [-np.inf, lo, hi, np.inf],
        labels=["low", "normal", "high"]
    ).astype(str)
    hour = x.index.tz_convert("America/New_York").hour
    minute = x.index.tz_convert("America/New_York").minute
    hm = hour * 60 + minute
    labels = np.select(
        [
            (hm >= 18*60) | (hm < 2*60),
            (hm >= 2*60) & (hm < 8*60),
            (hm >= 8*60) & (hm < 9*60+30),
            (hm >= 9*60+30) & (hm < 12*60),
            (hm >= 12*60) & (hm < 13*60+30),
            (hm >= 13*60+30) & (hm < 16*60),
        ],
        ["asia","london","ny_premarket","ny_am","ny_lunch","ny_pm"],
        default="postmarket"
    )
    x["session"] = labels
    x["tod_30m"] = (hm // 30).astype(int)
    return x
