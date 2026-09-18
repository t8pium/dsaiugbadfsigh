from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADE_FILE = ROOT / "results" / "ce_body" / "trades.pkl"
OUT = ROOT / "results" / "ce_body" / "body_depth_5pct.csv"

if not TRADE_FILE.exists():
    raise SystemExit(f"Missing {TRADE_FILE}. Run the CE/body study first.")

trades = pd.read_pickle(TRADE_FILE)
z = trades[
    (trades["mode"] == "qualifying")
    & trades["outcome_code"].isin([1, -1, 2])
    & trades["depth"].between(0, 1, inclusive="left")
].copy()

edges = np.arange(0, 1.000001, 0.05)
labels = [f"{int(a*100)}-{int(b*100)}%" for a, b in zip(edges[:-1], edges[1:])]
z["depth_5pct"] = pd.cut(z["depth"], bins=edges, labels=labels, include_lowest=True, right=False)

rows = []
for (tf, bucket), g in z.groupby(["timeframe", "depth_5pct"], observed=True):
    wins = (g["outcome_code"] == 1).astype(float)
    ambiguous = (g["outcome_code"] == 2).sum()
    rows.append({
        "timeframe": tf,
        "depth_band": str(bucket),
        "N": len(g),
        "wins": int(wins.sum()),
        "loss_or_ambiguous": int(len(g) - wins.sum()),
        "ambiguous": int(ambiguous),
        "win_rate": float(wins.mean()),
        "mean_R_conservative": float(np.nanmean(g["realized_R_conservative"])),
        "median_RR": float(np.nanmedian(g["rr"])),
        "median_width_ticks": float(np.nanmedian(g["width_ticks"])),
    })

out = pd.DataFrame(rows).sort_values(["timeframe", "depth_band"])
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

target = out[(out["timeframe"] == "4H") & (out["depth_band"] == "45-50%")]
print(f"Wrote {OUT}")
if len(target):
    print("\n4H 45-50%:")
    print(target.to_string(index=False))
else:
    print("\nNo 4H 45-50% row found in this run.")
