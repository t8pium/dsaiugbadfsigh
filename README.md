# Fair Value Gaps — Predictive Strength (MNQ)

Reproducible code for the [published FVG research report](https://t8pium.github.io/projects/fvg-predictive-strength/). The project tests whether mechanically defined Fair Value Gaps add information after controlling for distance, volatility, trend, session, and ordinary price revisits.

The published conclusion is deliberately modest: FVGs showed weak, short-lived, context-dependent structure, not persistent deterministic “magnetism” or an FVG-only trading edge.

## Fastest path: ZIP to Research Lab

1. [Download the repository ZIP](https://github.com/t8pium/fvg-predictive-strength/archive/refs/heads/main.zip).
2. Extract it anywhere, including a folder whose name contains spaces.
3. On Windows, double-click **`START_HERE.bat`**.
4. The launcher finds a supported 64-bit Python, creates `.fvg_venv`, installs the exact dependencies and local package, and opens the Streamlit Research Lab.
5. Open any of the nine experiment cards. Published methods, frozen reference evidence, charts, and source code are available without market data.
6. To reproduce, use **Data Setup** or an experiment’s **Reproduce** tab to upload licensed data or select data already on disk.

The supported interpreter range is 64-bit Python 3.11–3.13. Python 3.10 is not supported by this pinned snapshot. On macOS/Linux, use `./START_HERE.command` or `python3 bootstrap.py`.

## Data setup in the dashboard

Licensed Databento data is not redistributed. The UI offers three paths:

- Download `GLBX.MDP3` / `ohlcv-1m` / `MNQ.FUT` using your own API key. The key remains only in the child-process environment and is never persisted. Historical requests may be billable.
- Drag and drop one or more `.dbn`, `.dbn.zst`, `.parquet`, `.pq`, `.csv`, `.csv.gz`, `.csv.zst`, or `.zip` files. ZIPs may contain multiple data files and Databento symbology JSON sidecars.
- For multi-GB data, enter one or more local file/folder paths. This avoids copying the upload through the browser and streams chunks into a disk-backed staging database.

The importer resolves native DBN symbols with Databento’s mapping metadata, combines batch parts, rejects archive traversal/encryption/bombs, and builds both:

- `data/processed/mnq_active_1m.parquet`
- `data/processed/active_mnq.pkl` (the canonical scripts’ input)

Active construction keeps only strict quarterly MNQ outrights (`MNQH6`, `MNQZ25`, `MNQH2026`, etc.; never spreads), assigns the trade date at 18:00 America/New_York, chooses the highest total daily-volume contract, deduplicates overlaps, validates OHLC, and concatenates without back-adjustment.

Published input snapshot:

```text
active rows: 2,303,483
contracts: 27
start: 2020-01-01 23:00:00+00:00
end: 2026-07-10 20:59:00+00:00
duplicate timestamps: 0
missing OHLC: 0
```

Databento range ends are exclusive, so the downloader requests through `2026-07-11` to include all of July 10.

## Published reference versus local reproduction

The dashboard labels every frozen value **PUBLISHED REFERENCE**. A **LOCAL REPRODUCTION** check appears only after a successful experiment manifest proves that the displayed file was generated from the current processed dataset. Old output is not treated as a fresh run.

Local runs write CSV/JSON/PNG artifacts below `results/` and a manifest below `results/_runs/`. Failures and child output remain visible in the UI; full logs are stored in `results/_logs/`.

## Canonical research code

The original analysis files are preserved byte-for-byte under `src/original/`:

| Script | Experiment families |
|---|---|
| `fvg_final_fast.py` | Raw fill, detailed matched attraction, controls/regimes/logit, chronological robustness |
| `fvg_strength_one_tf.py` | Multi-timeframe attraction, age decay, continuation, first-touch reaction, robustness |
| `fvg_midpoint_reaction.py` | Exact midpoint / consequent-encroachment race |
| `fvg_ce_rejection_study.py` | Candle-body depth/CE study and trade-level outputs |

`scripts/run_original.py` verifies their SHA-256 hashes, rewrites only known legacy container-path constants in a temporary parsed copy, and records exactly which outputs changed. It never edits `src/original/`.

Read [the experiment definitions](docs/EXPERIMENTS.md) and [the scientific audit](docs/SCIENTIFIC_AUDIT.md) before interpreting results. The audit documents preserved right-censoring, holdout, clustering, roll, and exploratory-multiplicity limitations rather than silently changing published methodology.

## Command-line reproduction

The launcher is the normal path. For an existing Python environment:

```bash
python -m pip install -e .
python scripts/prepare_active_contract.py --input YOUR_BATCH.zip
python scripts/run_original.py detailed-1m
python scripts/run_original.py multi-tf --tf 1
python scripts/run_original.py midpoint --tf 1
python scripts/run_original.py ce-body --ce-tfs 60,120,240
```

Use repeated `--input` arguments for multiple files. A directory is also accepted. To run every canonical family, use `python run_all.py`.

## Repository map

- `START_HERE.bat`, `START_HERE.command`, `bootstrap.py` — one-click bootstrapping.
- `dashboard.py` — nine-card local Research Lab.
- `fvg_research/` — portable IO, active-contract, detector, matching, bars, outcomes, and dashboard helpers.
- `scripts/` — download, preparation, canonical runner, CE postprocessing, and CI policy.
- `src/original/` — immutable published analysis scripts.
- `reference_results/reference_metrics.json` — frozen published evidence.
- `tests/` — synthetic package, detector, IO, DBN mapping, active-contract, UI, wrapper, and postprocessing tests.
- `data/` and `results/` — ignored local inputs/outputs (only their README files are tracked).

## Verification

```bash
python -m pip check
python -m compileall -q .
python -m unittest discover -s tests -v
python scripts/ci_policy_check.py
```

CI runs the install/import/compile/test/policy suite on Ubuntu (Python 3.11 and 3.13) and Windows (Python 3.12). It also rejects canonical hash drift, legacy container-path leakage outside `src/original`, and committed licensed/generated data.

## Important limits

- Browser upload is capped at 1 GiB; use the local-path importer for larger data.
- One-minute OHLC cannot reveal TP/SL ordering inside the same minute. The CE study conservatively counts same-minute ambiguity as a loss.
- Vendor history corrections can produce small differences from the frozen reference.
- Matching reduces obvious confounding but does not establish causality.
- The 4H 45–50% CE result is exploratory, not independently confirmed.
