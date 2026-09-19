from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize_bands(trades: pd.DataFrame) -> pd.DataFrame:
    required = {"mode", "outcome_code", "depth", "timeframe", "realized_R_conservative", "rr", "width_ticks"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Trade file is missing columns: {sorted(missing)}")
    sample = trades.loc[
        (trades["mode"] == "qualifying")
        & trades["outcome_code"].isin([1, -1, 2])
        & (trades["depth"] >= 0)
        & (trades["depth"] < 1)
    ].copy()
    edges = np.arange(0, 1.000001, 0.05)
    labels = [f"{int(round(low * 100))}-{int(round(high * 100))}%" for low, high in zip(edges[:-1], edges[1:])]
    sample["depth_band"] = pd.cut(
        sample["depth"], bins=edges, labels=labels, include_lowest=True, right=False
    )
    rows = []
    for (timeframe, band), group in sample.groupby(["timeframe", "depth_band"], observed=True):
        wins = group["outcome_code"].eq(1)
        ambiguous = group["outcome_code"].eq(2)
        rows.append({
            "timeframe": timeframe, "depth_band": str(band), "N": len(group),
            "wins": int(wins.sum()), "loss_or_ambiguous": int((~wins).sum()),
            "ambiguous": int(ambiguous.sum()), "win_rate": float(wins.mean()),
            "mean_R_conservative": float(group["realized_R_conservative"].mean()),
            "median_RR": float(group["rr"].median()),
            "median_width_ticks": float(group["width_ticks"].median()),
            "depth_low": float(edges[labels.index(str(band))]),
        })
    columns = [
        "timeframe", "depth_band", "N", "wins", "loss_or_ambiguous", "ambiguous",
        "win_rate", "mean_R_conservative", "median_RR", "median_width_ticks", "depth_low",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["timeframe", "depth_low"]).reset_index(drop=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild conservative 5%-depth CE/body bands from a local trade file.")
    parser.add_argument("--trade-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.trade_file.is_file():
        print(f"ERROR: Trade file does not exist: {args.trade_file}", file=sys.stderr)
        return 2
    try:
        output = summarize_bands(pd.read_pickle(args.trade_file))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    target = output.loc[(output["timeframe"] == "4H") & (output["depth_band"] == "45-50%")]
    if len(target):
        print(target.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
