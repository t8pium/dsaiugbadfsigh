from __future__ import annotations

import numpy as np
import pandas as pd


def rate(values) -> float:
    series = pd.Series(values).dropna()
    return float(series.mean()) if len(series) else np.nan


def diff_pp(a, b) -> float:
    return 100.0 * (rate(a) - rate(b))


def chronological_split(df: pd.DataFrame, frac: float = 0.70):
    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1")
    ordered = df.sort_index()
    cutoff = int(len(ordered) * frac)
    return ordered.iloc[:cutoff], ordered.iloc[cutoff:]


def cluster_bootstrap_diff(frame: pd.DataFrame, a: str, b: str, day_col: str,
                           n_boot: int = 1000, seed: int = 20260918):
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    days = pd.Index(frame[day_col].dropna().unique())
    if days.empty:
        return np.array([np.nan, np.nan, np.nan])
    values = []
    groups = {day: frame.loc[frame[day_col] == day] for day in days}
    for _ in range(n_boot):
        sampled = rng.choice(days, size=len(days), replace=True)
        combined = pd.concat([groups[day] for day in sampled], ignore_index=True)
        values.append((combined[a].mean() - combined[b].mean()) * 100)
    return np.percentile(values, [2.5, 50, 97.5])
