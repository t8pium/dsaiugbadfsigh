from __future__ import annotations
import numpy as np
import pandas as pd

def rate(x) -> float:
    s = pd.Series(x).dropna()
    return float(s.mean()) if len(s) else np.nan

def diff_pp(a, b) -> float:
    return 100.0 * (rate(a) - rate(b))

def chronological_split(df: pd.DataFrame, frac: float = 0.70):
    x = df.sort_index()
    cut = int(len(x) * frac)
    return x.iloc[:cut], x.iloc[cut:]

def cluster_bootstrap_diff(frame: pd.DataFrame, a: str, b: str, day_col: str,
                           n_boot: int = 1000, seed: int = 20260918):
    rng = np.random.default_rng(seed)
    days = pd.Index(frame[day_col].dropna().unique())
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(days, size=len(days), replace=True)
        chunks = [frame[frame[day_col] == d] for d in sampled]
        z = pd.concat(chunks, ignore_index=True)
        vals.append((z[a].mean() - z[b].mean()) * 100)
    return np.percentile(vals, [2.5, 50, 97.5])
