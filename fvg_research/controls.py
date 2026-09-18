from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

FEATURES = ["distance_atr", "width_atr", "body_b_atr"]
CATS = ["direction", "session", "tod_30m", "vol_regime", "trend"]

def candidate_zones(df_state: pd.DataFrame, fvg_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build non-FVG candidate timestamps; zone geometry is assigned after matching."""
    c = df_state.loc[~df_state.index.isin(fvg_index)].copy()
    c["direction"] = np.sign(c["ret20"]).replace(0, 1).astype(int)
    c["distance_atr"] = np.nan
    c["width_atr"] = np.nan
    c["body_b_atr"] = (c["close"] - c["open"]).abs() / c["atr14"]
    return c

def matched_controls(events: pd.DataFrame, state: pd.DataFrame, n_controls: int = 1,
                     seed: int = 20260918, max_events: int | None = None) -> pd.DataFrame:
    """Match non-FVG timestamps within exact state strata and nearest continuous features.

    Control zones copy the event's direction, width and ATR-normalized distance, then
    are placed relative to the control timestamp's close. This preserves geometry
    while changing the timestamp/location source.
    """
    rng = np.random.default_rng(seed)
    e = events.copy()
    if max_events and len(e) > max_events:
        e = e.iloc[np.sort(rng.choice(len(e), max_events, replace=False))]
    e = e.join(state[["session","tod_30m","vol_regime","trend"]], how="left")
    candidates = state.loc[~state.index.isin(events.index)].dropna(
        subset=["atr14","session","tod_30m","vol_regime","trend"]
    ).copy()

    out = []
    for key, eg in e.groupby(CATS, dropna=False):
        mask = np.ones(len(candidates), dtype=bool)
        for col, val in zip(CATS, key if isinstance(key, tuple) else (key,)):
            mask &= candidates[col].to_numpy() == val
        cg = candidates.loc[mask]
        if len(cg) < max(10, n_controls):
            continue

        # Continuous matching is dominated by time-local body/vol state. Geometry is copied exactly.
        x_c = np.column_stack([
            (cg["close"] - cg["open"]).abs().div(cg["atr14"]).fillna(0).to_numpy(),
            cg["volatility"].fillna(0).to_numpy() * 1000,
        ])
        x_e_state = state.reindex(eg.index)
        x_e = np.column_stack([
            eg["body_b_atr"].fillna(0).to_numpy(),
            x_e_state["volatility"].fillna(0).to_numpy() * 1000,
        ])
        nn = NearestNeighbors(n_neighbors=min(n_controls, len(cg))).fit(x_c)
        _, inds = nn.kneighbors(x_e)
        for row_i, neigh in enumerate(inds):
            ev = eg.iloc[row_i]
            for j in neigh:
                ts = cg.index[j]
                bar = cg.iloc[j]
                atr = bar["atr14"]
                dist = ev["distance_atr"] * atr
                width = ev["width_atr"] * atr
                if ev["direction"] == 1:
                    upper = bar["close"] - dist
                    lower = upper - width
                    near, far = upper, lower
                else:
                    lower = bar["close"] + dist
                    upper = lower + width
                    near, far = lower, upper
                out.append({
                    "event_ts": ev.name, "control_ts": ts,
                    "direction": int(ev["direction"]),
                    "lower": lower, "upper": upper, "near": near, "far": far,
                    "mid": (lower + upper)/2, "width": width,
                    "width_atr": ev["width_atr"], "distance_atr": ev["distance_atr"],
                    "session": bar["session"], "tod_30m": int(bar["tod_30m"]),
                    "vol_regime": bar["vol_regime"], "trend": int(bar["trend"]),
                })
    return pd.DataFrame(out)
