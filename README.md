# Fair Value Gaps — Predictive Strength (MNQ) — Reproducibility

This directory contains the code behind the portfolio project **“Do Fair Value Gaps predict price?”** The goal is reproducibility, not persuasion: a reviewer should be able to obtain the same MNQ data, rebuild the same active-contract series, rerun the experiments, and inspect the generated tables.

**Published conclusion:** FVGs showed weak, short-lived, context-dependent predictive structure. The evidence did not support persistent deterministic “magnetism” or an FVG-only trading edge.

## Repository map

- `prepare_data.py` — rebuilds the exact active MNQ one-minute series from the Databento parent-symbol export.
- `run_published.py` — runs the published experiment suite.
- `published/fvg_final_fast.py` — deep 1m fill / matched-control / directional / stratification / logistic / sensitivity / holdout analysis.
- `published/fvg_strength_one_tf.py` — multi-timeframe attraction, age decay, formation continuation, first-touch reaction, and 5-bar holdout.
- `published/fvg_midpoint_reaction.py` — exact midpoint / consequent-encroachment matched race.
- `published/fvg_ce_rejection_study.py` — candle-body acceptance / CE execution study from 1m through daily.
- `reference_results/` — frozen small CSV outputs from the published run.
- `tests/` — synthetic tests that require no licensed market data.

The portable scripts preserve the published algorithms, seeds, horizons, matching rules, and ambiguity handling. Only input/output path configuration was changed from the original research-container scripts.

## Install

```bash
git clone https://github.com/t8pium/t8pium.github.io.git
cd t8pium.github.io/reproducibility/fvg-predictive-strength
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt\npip install -e .
```

Python 3.11+ is recommended.

## Obtain the data

The raw historical market data is licensed and is **not redistributed** here.

The published study used a Databento request with:

- dataset: `GLBX.MDP3`
- parent symbol: `MNQ.FUT`
- input symbology: `parent`
- schema: `ohlcv-1m`
- coverage: 2020-01-01 through 2026-07-10

Original study archive SHA-256:

```text
6a9d150999fb846abb00f202816ca2cc10051a79ad29a94a517b668a6e03a74c
```

Build the active series:

```bash
python prepare_data.py --input /path/to/GLBX-20260711-3WDB9VAAF6.zip
```

Expected snapshot:

```text
active rows: 2,303,483
contracts: 27
start: 2020-01-01 23:00:00+00:00
end: 2026-07-10 20:59:00+00:00
duplicate timestamps: 0
missing OHLC: 0
```

The parent-symbol export contains multiple listed contracts at the same timestamp. `prepare_data.py` keeps quarterly MNQ outrights and selects the highest-total-volume outright for each CME trading date. The CME trading date advances at 18:00 America/New_York. Prices are unadjusted.

## Mechanical FVG definition

For completed candles A=t−2, B=t−1, C=t:

```text
Bullish FVG: Low[C]  > High[A]
Bearish FVG: High[C] < Low[A]
```

Bullish zone = `High[A] -> Low[C]`.
Bearish zone = `High[C] -> Low[A]`.

An FVG becomes known only **after candle C closes**. The experiments never act as though the gap existed earlier.

## Run

```bash
python run_published.py --suite all
```

Individual suites:

```bash
python run_published.py --suite deep-1m
python run_published.py --suite multitimeframe
python run_published.py --suite midpoint
python run_published.py --suite body
```

## Experiment map

### 01 — Raw fill rates
Canonical code: `published/fvg_final_fast.py`.

Measures touch, 50% mitigation, full fill, and eventual touch. Fixed one-minute horizons are 5, 15, 30, 60, 120, 240, 1,380, and 4,140 trading bars. The result is interpreted only relative to matched ordinary zones.

### 02 — Matched-zone attraction
Canonical code: `fvg_final_fast.py` and `fvg_strength_one_tf.py`.

Deep 1m study: RNG seed 42, up to 40,000 real FVGs, 5 controls per FVG. Multi-timeframe study: seed 260918, up to 12,000 FVGs per timeframe, 3 controls per FVG.

Controls preserve direction and ATR-normalized width/distance and match session, volatility regime, trend regime, and time bucket.

### 03 — Age decay
Canonical code: `fvg_strength_one_tf.py`.

Conditional transitions: 1→3, 3→5, 5→10, 10→20 native bars. The test asks whether a gap that has survived unfilled is still unusually likely to be touched next.

### 04 — Formation continuation
Canonical code: `fvg_strength_one_tf.py`.

FVG moves are matched to non-FVG moves by session, volatility/trend regime, time bucket, signed displacement direction, 3-bar movement/ATR bin, and middle-candle body/ATR bin. Forward directional returns are tested at 1, 3, 5, and 10 bars.

### 05 — First-touch reaction
Canonical code: `fvg_strength_one_tf.py`, function `first_touch_reaction`.

Near-edge touch must occur within 20 native bars. Starting the next bar, price races a rejection target one full gap width away against the far edge for 10 bars. Same-bar target+far hits are ambiguous.

### 06 — Midpoint / CE
Canonical code: `fvg_midpoint_reaction.py`; RNG seed 9917.

The exact midpoint must be touched within 20 bars. Starting on the next bar, the equidistant near edge and far edge race for 10 bars. Same-bar hits are ambiguous. Three matched controls per FVG; day-cluster bootstrap uses 500 repetitions.

### 07 — Candle-body acceptance around CE
Canonical code: `fvg_ce_rejection_study.py`.

Timeframes: 1m, 2m, 3m, 5m, 10m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D.

A later opposite-colored candle must wick into the FVG and body-close inside it before a body-close invalidation. Depth is normalized 0%=near edge, 50%=CE, 100%=far edge. Entry is the signal close; stop is the far edge; target is the near edge. Higher-TF signals are resolved on future 1m bars. Same-minute TP+SL is ambiguous and is a loss only in the explicitly labeled conservative expectancy.

The 4H 45–50% result is **hypothesis-generating**, not confirmed.

### 08 — Controls, distance and regimes
Canonical code: `fvg_final_fast.py`.

The 60m outcome is stratified by distance, size, session, direction, volatility, trend, displacement and year. The logistic model uses FVG status, distance, width, volatility ratio, trend score, cyclical time-of-day terms, and direction, with `LogisticRegression(max_iter=200, C=1e6)`.

### 09 — Out-of-sample robustness
Canonical code: all three relevant programs.

The time series is split chronologically, never randomly shuffled. The multi-timeframe magnet test uses a 70/30 split of matched events. The deep 1m study separately tests 60m attraction by chronological split and by year. The CE/body study applies its own chronological 70/30 split to the 45–55% subset.

## Frozen random seeds

- deep 1m controls: 42
- deep 1m cluster bootstrap: 7
- multi-timeframe experiments: 260918
- midpoint experiment: 9917
- CE/body bootstrap: 20260918

## Important limitations

- One primary market: MNQ.
- One-minute OHLCV cannot reveal TP/SL ordering inside the same minute.
- Matched controls reduce obvious confounding but do not prove causality.
- Related FVGs from the same move/day are not independent; day clustering is used where practical.
- Many timeframe/depth cells were inspected, so post-hoc strong subgroups need independent replication.
- Statistical significance is not equivalent to net trading profitability.

Portfolio report: https://t8pium.github.io/projects/fvg-predictive-strength/


## Original-script integrity

The files in `src/original/` are the study scripts retained from the original analysis run. They intentionally still show the original `/mnt/data` paths. Do not edit them if you want an auditable copy.

`scripts/run_original.py` reads an original script, patches only those environment-specific paths into a temporary copy, and executes the temporary file. Random seeds and experiment logic remain unchanged.

## Expected compute

The one-minute detailed study and CE body-acceptance study are the heaviest. Runtime depends strongly on CPU, available RAM, and whether Numba has already compiled its functions. Multi-timeframe runs are designed to be executed one timeframe at a time.

## Data licensing

This repository does not redistribute Databento market data. The downloader is provided so a reader with their own Databento access can recreate the source bars. Databento documents parent futures requests with `stype_in="parent"` and one-minute OHLCV with `schema="ohlcv-1m"`.
