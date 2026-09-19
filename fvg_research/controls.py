from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

STATE_COLUMNS = ["session", "tod_30m", "vol_regime", "trend"]
OUTPUT_COLUMNS = [
    "event_ts", "control_ts", "direction", "lower", "upper", "near", "far", "mid",
    "width", "width_atr", "distance_atr", *STATE_COLUMNS,
]


def candidate_zones(state: pd.DataFrame, fvg_index: pd.DatetimeIndex) -> pd.DataFrame:
    candidates = state.loc[~state.index.isin(fvg_index)].copy()
    candidates["body_b_atr"] = (candidates["close"] - candidates["open"]).abs() / candidates["atr14"]
    return candidates


def matched_controls(events: pd.DataFrame, state: pd.DataFrame, n_controls: int = 1,
                     seed: int = 20260918, max_events: int | None = None) -> pd.DataFrame:
    if n_controls < 1:
        raise ValueError("n_controls must be >= 1")
    rng = np.random.default_rng(seed)
    event_frame = events.copy()
    if max_events and len(event_frame) > max_events:
        event_frame = event_frame.iloc[np.sort(rng.choice(len(event_frame), max_events, replace=False))]
    # Events produced by experiments/common already contain state columns. Replace
    # them explicitly so join overlap never raises and state has one source of truth.
    event_frame = event_frame.drop(columns=STATE_COLUMNS, errors="ignore").join(
        state[STATE_COLUMNS], how="left"
    )
    candidates = candidate_zones(state, events.index).dropna(
        subset=["atr14", "volatility", *STATE_COLUMNS]
    )
    output: list[dict[str, object]] = []
    for key, group in event_frame.dropna(subset=STATE_COLUMNS).groupby(STATE_COLUMNS, dropna=False):
        mask = np.ones(len(candidates), dtype=bool)
        for column, value in zip(STATE_COLUMNS, key):
            mask &= candidates[column].to_numpy() == value
        pool = candidates.loc[mask]
        if len(pool) < max(10, n_controls):
            continue
        candidate_features = np.column_stack([
            pool["body_b_atr"].fillna(0).to_numpy(),
            pool["volatility"].fillna(0).to_numpy() * 1000,
        ])
        event_state = state.reindex(group.index)
        event_features = np.column_stack([
            group["body_b_atr"].fillna(0).to_numpy(),
            event_state["volatility"].fillna(0).to_numpy() * 1000,
        ])
        neighbors = NearestNeighbors(n_neighbors=min(n_controls, len(pool))).fit(candidate_features)
        _, indices = neighbors.kneighbors(event_features)
        for row_number, matches in enumerate(indices):
            event = group.iloc[row_number]
            for match in matches:
                timestamp = pool.index[match]
                bar = pool.iloc[match]
                scale = bar["atr14"]
                distance = event["distance_atr"] * scale
                width = event["width_atr"] * scale
                if event["direction"] == 1:
                    upper, lower = bar["close"] - distance, bar["close"] - distance - width
                    near, far = upper, lower
                else:
                    lower, upper = bar["close"] + distance, bar["close"] + distance + width
                    near, far = lower, upper
                output.append({
                    "event_ts": event.name, "control_ts": timestamp,
                    "direction": int(event["direction"]), "lower": lower, "upper": upper,
                    "near": near, "far": far, "mid": (lower + upper) / 2, "width": width,
                    "width_atr": event["width_atr"], "distance_atr": event["distance_atr"],
                    "session": bar["session"], "tod_30m": int(bar["tod_30m"]),
                    "vol_regime": bar["vol_regime"], "trend": int(bar["trend"]),
                })
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)
