# Experiment documentation

This document maps every public portfolio experiment to the exact original analysis code and describes the experiment as an executable sequence.

## Shared input

All experiments use the active-contract one-minute MNQ series created by:

```bash
python scripts/download_databento.py
python scripts/prepare_active_contract.py
```

The prepared `active_mnq.pkl` contains at minimum:

- `ts_event` — UTC timestamp
- `open`, `high`, `low`, `close`
- `volume`
- `symbol` — listed MNQ contract
- `trade_date`

The active contract for each CME trading date is the listed expiry with the highest total one-minute volume for that trading date.

The original study excludes event formation within approximately one day of an active-contract transition.

---

## Experiment 01 — Raw fill rates

**Canonical code:** `src/original/fvg_final_fast.py`

### Event construction

For each one-minute bar C after sufficient warmup:

- bullish if `low[C] > high[C-2]`
- bearish if `high[C] < low[C-2]`
- bullish lower/far edge = `high[C-2]`
- bullish upper/near edge = `low[C]`
- bearish lower/near edge = `high[C]`
- bearish upper/far edge = `low[C-2]`
- midpoint = mean of lower and upper

Events require valid ATR, trend/volatility state, positive width, non-negative distance from current close, and no roll-block flag.

### Outcomes

Future rolling extrema are computed for:

`5, 15, 30, 60, 120, 240, 1380, 4140` one-minute bars.

For each horizon, the script records:

- near-edge touch
- midpoint / 50% mitigation
- far-edge / full fill

It also builds future suffix minima/maxima to determine whether each FVG is ever touched or fully filled later in the available sample.

### Run

```bash
python scripts/run_original.py detailed-1m
```

The output contains `summary.json`, `main_results.csv`, `fvg_events.csv`, and plots.

---

## Experiment 02 — Matched-zone attraction

**Canonical code:** both
- `src/original/fvg_final_fast.py` for the deep 1m placebo test
- `src/original/fvg_strength_one_tf.py` for multi-timeframe tests

### Deep 1m control construction

1. Randomly sample up to 40,000 eligible FVGs with seed 42.
2. Build candidate bars that are **not FVG formation bars** and have valid ATR/trend/volatility state.
3. Exact-match candidates by:
   - session
   - hour-of-day bucket
   - volatility regime
   - trend regime
4. For each selected candidate, copy the real FVG's:
   - direction
   - ATR-normalized distance
   - ATR-normalized width
5. Convert those normalized values into absolute control-zone geometry using the candidate bar's ATR and close.
6. Create five controls per FVG where candidate availability permits.

This is crucial: the control is not a random horizontal line anywhere on the chart. It is designed to be similarly distant and similarly sized under a similar market state.

### Outcome comparison

Real FVG and control touch probabilities are evaluated over identical future horizons. Control outcomes are averaged at the parent-FVG level before paired differences are calculated.

Day-cluster bootstrap samples entire trading-day groups to estimate uncertainty around the paired effect.

### Multi-timeframe version

For 1m, 5m, 15m, 1H, and 4H:

- sample up to 12,000 FVGs per timeframe
- generate three controls per event
- use native-bar horizons 1, 3, 5, 10, 20
- anchor resampling to the CME 18:00 ET session

### Run

```bash
python scripts/run_original.py detailed-1m

for tf in 1 5 15 60 240; do
  python scripts/run_original.py multi-tf --tf $tf
done
```

---

## Experiment 03 — FVG age decay

**Canonical code:** `src/original/fvg_strength_one_tf.py`

This is a conditional survival-style test.

For each matched FVG/control set:

1. Evaluate whether the zone has been touched by bar 1.
2. Among zones that survived untouched, ask whether they become touched by bar 3.
3. Repeat for 3→5, 5→10, and 10→20.
4. Compare the conditional FVG touch rate with the conditional control touch rate.

This differs from the ordinary cumulative fill-rate test because an FVG that has already filled is removed from the later-age question.

Run with the same multi-timeframe commands as Experiment 02.

---

## Experiment 04 — Formation continuation

**Canonical code:** `src/original/fvg_strength_one_tf.py`

This experiment asks whether the **move that created the FVG** predicts continuation.

### Non-FVG move matching

Candidate non-FVG bars are matched to FVG formation events on discrete bins of:

- session
- volatility regime
- trend regime
- time-of-day bucket
- move direction
- three-bar move magnitude in ATR
- middle-candle body magnitude in ATR

Up to 8,000 FVG events per timeframe are sampled. One matched non-FVG move is selected from the exact state bucket.

Forward returns are measured after 1, 3, 5, and 10 native bars, signed in the original formation direction and normalized by ATR.

A positive FVG-minus-control difference means the FVG-forming move continued more strongly than a similar non-FVG displacement.

---

## Experiment 05 — First-touch retest reaction

**Canonical code:** `src/original/fvg_strength_one_tf.py`

For each real FVG and matched control zone:

1. Search up to 20 native bars for the first near-edge touch.
2. Once touched, define:
   - far edge = full traversal of the zone
   - rejection target = one full zone width away from the near edge, in the rejection direction
3. Starting **after the touch bar**, scan up to 10 native bars.
4. Record:
   - rejection target first = win
   - far edge first = loss
   - both inside the same native OHLC bar = ambiguous
   - neither = censored
5. Separately measure normalized close displacement three bars after touch.

Control race success is summarized at the parent-FVG level before comparison.

---

## Experiment 06 — Midpoint / CE race

**Canonical code:** `src/original/fvg_midpoint_reaction.py`

### Matching

Up to 12,000 FVGs per timeframe are sampled. Three controls per event are created using the same state-matched geometry logic as the multi-timeframe attraction study.

### Midpoint event

For each FVG/control:

1. Search up to 20 native bars for the exact midpoint to be traded.
2. If midpoint is never reached, the case is not resolved.
3. From the **next bar**, race:
   - near edge
   - far edge
4. Because both boundaries are equally distant from the midpoint, neither outcome has a geometric distance advantage.
5. If one OHLC bar contains both boundaries, classify it as ambiguous.

### Statistics

Resolved real-vs-parent-control differences are bootstrapped by trading day (500 replications in the original script). Year-specific rows are also emitted.

### Run

```bash
for tf in 1 5 15 60 240; do
  python scripts/run_original.py midpoint --tf $tf
done
```

---

## Experiment 07 — Candle-body acceptance around CE

**Canonical code:** `src/original/fvg_ce_rejection_study.py`

Timeframes:

`1m, 2m, 3m, 5m, 10m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D`.

Higher-timeframe bars are built from the active one-minute series and anchored to the CME session.

### Trigger search

For each FVG:

- bullish FVG requires a later bearish candle
- bearish FVG requires a later bullish candle
- the candle's wick must touch the gap
- its close must remain inside the gap
- wick length is otherwise irrelevant
- search stops if a prior candle **body closes beyond the far edge**

Close depth is:

- 0% = near edge
- 50% = midpoint / CE
- 100% = far edge

The study records both:

- first qualifying opposite candle
- stricter first-raw-touch sensitivity

### Trade execution

Entry = qualifying candle close.

Target = near FVG edge.

Stop = far FVG edge.

Every trade is resolved on the underlying **one-minute bars**, not the higher-timeframe candle.

Outcome codes:

- +1 = target first
- -1 = stop first
- +2 = target and stop both inside the same one-minute bar
- 0 = censored before the active contract segment ends

Conservative results count +2 ambiguity as a loss. The raw ambiguity count is also reported.

### Depth studies

The script calculates:

- 10%-wide penetration buckets
- exact 50%
- 49–51%
- 47.5–52.5%
- 45–55%
- 40–60%
- year results
- direction results
- chronological 70/30 results
- first-touch sensitivity
- trade-level bootstrap confidence intervals

### Run

```bash
python scripts/run_original.py ce-body
```

---

## Experiment 08 — Controls, distance and regimes

**Canonical code:** `src/original/fvg_final_fast.py`

The detailed 1m script stratifies the 60-minute matched touch experiment by:

- ATR-normalized distance
- ATR-normalized FVG width
- session
- bullish/bearish direction
- volatility regime
- trend regime
- displacement body size
- calendar year

Minimum-size sensitivity includes:

- any gap
- ≥1 tick
- ≥2 ticks
- ≥0.05 ATR
- ≥0.10 ATR
- ≥0.20 ATR

### Logistic model

The combined FVG/control rows are modeled with a logistic regression predicting 60-minute touch.

Features:

- `is_fvg`
- `distance_atr`
- `width_atr`
- volatility ratio
- trend score
- direction
- cyclical time-of-day sine/cosine

The key coefficient is the FVG indicator; `exp(coef)` is reported as the FVG odds ratio.

---

## Experiment 09 — Chronological out-of-sample robustness

**Canonical code:** both `fvg_final_fast.py` and `fvg_strength_one_tf.py`.

### Multi-timeframe holdout

The matched sample is sorted chronologically. The first 70% and last 30% are compared separately for the five-native-bar attraction outcome.

### Detailed 1m holdout

The 40k matched 1m sample uses the timestamp 70th percentile as the chronological cutoff. The 60-minute FVG/control effect is recomputed independently in the early and late periods.

### Year stability

The same detailed script reports matched 60-minute differences by calendar year.

No random train/test shuffle is used.

---

## Why the code is split across four original scripts

The project evolved experimentally rather than being designed as a library first. The original scripts were optimized for the heavy computations actually being run:

- one detailed 1m falsification script,
- one parameterized multi-timeframe script,
- one midpoint-specific script,
- one execution-specific CE script.

They are preserved unchanged for auditability. The portable runner modifies only paths in a temporary copy.

## What is *not* claimed

Re-running the code on a newer Databento correction, a different roll construction, or a different time period can produce different numbers. Reproducibility here means the experiment definitions and code are public and executable—not that a market effect is guaranteed to remain numerically identical forever.
