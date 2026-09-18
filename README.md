# Fair Value Gaps — Predictive Strength (MNQ)

Reproducible code for the portfolio research project:

**https://t8pium.github.io/projects/fvg-predictive-strength/**

The project asks whether mechanically defined Fair Value Gaps contain incremental information about future MNQ price behavior after controlling for distance, volatility, trend, session, and ordinary price revisits.

## One-click interactive research lab

If you just want to inspect the project visually:

1. **[Download the repository ZIP](https://github.com/t8pium/fvg-predictive-strength/archive/refs/heads/main.zip)**.
2. Extract the ZIP.
3. On Windows, double-click **`START_HERE.bat`**.
4. The launcher creates an isolated Python environment and installs the required libraries automatically.
5. A local browser dashboard opens with one card for every experiment.

macOS/Linux users can run `START_HERE.command` or `python3 bootstrap.py`.

The dashboard separates two things clearly:

- **Published evidence** — frozen reference tables/charts from the original study.
- **Local reproduction** — outputs produced by the experiment code on the reviewer's own machine.

The licensed MNQ market data is not embedded in the ZIP. From the dashboard's **Data Setup** page, a reviewer can either use their own Databento API key or **drag and drop the market-data file directly into the app**. Direct upload supports Databento `.dbn` / `.dbn.zst`, Parquet, CSV / CSV.GZ, ZIP archives, and multiple batch files selected together. Downloading historical vendor data may be billable under the reviewer's Databento plan.

**Published conclusion:** FVGs showed weak, short-lived, context-dependent predictive structure. The evidence did not support persistent deterministic “magnetism” or an FVG-only trading edge.

## Repository structure

- `src/original/` — exact analysis scripts retained from the original study run.
- `scripts/download_databento.py` — recreates the raw parent-symbol one-minute OHLCV input.
- `scripts/prepare_active_contract.py` — rebuilds the active MNQ one-minute series used by the study.
- `scripts/run_original.py` — runs the original scripts portably by changing only environment-specific file paths in a temporary copy.
- `START_HERE.bat` — Windows one-click launcher; creates a private environment, installs dependencies, and opens the dashboard.
- `dashboard.py` — interactive experiment cards, published charts, source viewer, reproduction controls, and local-output browser.
- `bootstrap.py` — dependency/bootstrap logic used by the one-click launcher.
- `reference_results/` — frozen published metrics displayed by the dashboard.
- `run_all.py` — executes the complete published experiment family.
- `fvg_research/` — shared reusable helpers.
- `docs/EXPERIMENTS.md` — exact construction of every experiment.
- `results/` — generated experiment outputs.
- `data/` — local raw/processed data locations; licensed market data is not committed.

## Install

```bash
git clone https://github.com/t8pium/fvg-predictive-strength.git
cd fvg-predictive-strength

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Python 3.10+ is required.

## Obtain the market data

### Direct file upload

The easiest offline path is inside the local Research Lab:

1. Double-click `START_HERE.bat`.
2. Open **Data Setup**.
3. Under **Option B — Upload your own Databento file**, drag in one or more files.
4. Click **Import file(s) + build active MNQ dataset**.

The same uploader also appears inside an experiment's **Reproduce** tab whenever the dataset is missing.

Supported uploads:

- Databento `.dbn` and `.dbn.zst`
- `.parquet` / `.pq`
- `.csv` / `.csv.gz`
- `.zip` containing any supported file type
- multiple files from a Databento batch download

Large local uploads are allowed up to 4 GB by the bundled Streamlit configuration.


The raw historical market data is licensed and is **not redistributed**.

The study used:

- Databento dataset: `GLBX.MDP3`
- schema: `ohlcv-1m`
- input symbology: `parent`
- parent symbol: `MNQ.FUT`
- period: 2020-01-01 through 2026-07-10

With your own Databento key:

```bash
export DATABENTO_API_KEY="YOUR_KEY"
python scripts/download_databento.py
python scripts/prepare_active_contract.py
```

The preparation script reproduces the published active-series construction:

1. keep quarterly MNQ outright contracts only: `MNQ[HMUZ][0-9]`;
2. assign CME trade date using America/New_York, with bars at or after 18:00 ET assigned to the following date;
3. sum total one-minute volume by listed contract for each trade date;
4. select the highest-volume outright for that trading date;
5. concatenate the selected bars without back-adjusting prices.

Expected published snapshot:

```text
active rows: 2,303,483
contracts: 27
start: 2020-01-01 23:00:00+00:00
end: 2026-07-10 20:59:00+00:00
duplicate timestamps: 0
missing OHLC: 0
```

## Mechanical FVG definition

For completed candles A = t−2, B = t−1, C = t:

```text
Bullish FVG: Low[C]  > High[A]
Bearish FVG: High[C] < Low[A]
```

Bullish zone = `High[A] → Low[C]`.

Bearish zone = `High[C] → Low[A]`.

An FVG becomes known only **after candle C closes**.

## Run the published experiment suite

```bash
python run_all.py
```

Individual canonical suites:

```bash
# Deep one-minute study:
# raw fill, matched controls, stratification, logistic model, sensitivity, OOS
python scripts/run_original.py detailed-1m

# Multi-timeframe attraction, age decay, continuation, retest reaction, OOS
python scripts/run_original.py multi-tf --tf 1
python scripts/run_original.py multi-tf --tf 5
python scripts/run_original.py multi-tf --tf 15
python scripts/run_original.py multi-tf --tf 60
python scripts/run_original.py multi-tf --tf 240

# Midpoint / consequent encroachment
python scripts/run_original.py midpoint --tf 1
python scripts/run_original.py midpoint --tf 5
python scripts/run_original.py midpoint --tf 15
python scripts/run_original.py midpoint --tf 60
python scripts/run_original.py midpoint --tf 240

# Candle-body / CE acceptance study from 1m through daily
python scripts/run_original.py ce-body
```

## Canonical source map

| Script | Experiment families |
|---|---|
| `src/original/fvg_final_fast.py` | Raw fill rates, detailed 1m matched attraction, directional movement, distance/size/session/volatility/trend/displacement/year stratification, sensitivity, logistic model, 70/30 holdout |
| `src/original/fvg_strength_one_tf.py` | Multi-timeframe attraction, age decay, formation continuation, first-touch retest reaction, 5-bar chronological holdout |
| `src/original/fvg_midpoint_reaction.py` | Exact midpoint / CE symmetric race with matched controls and yearly breakdown |
| `src/original/fvg_ce_rejection_study.py` | Body-close penetration bands, exact 50%, 45–55%, first-touch sensitivity, 1m→daily execution, exploratory 4H 45–50% result |

For exact matching rules, horizons, censoring, ambiguity handling, random seeds, and validation design, read:

**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**

## Original-script integrity

The files under `src/original/` are the analysis scripts preserved from the original research run. They intentionally retain the original container paths.

`scripts/run_original.py` loads an original script, changes only machine-specific input/output paths in a temporary copy, and executes that temporary file. The original files are not modified.

## Important limitations

- The primary market is MNQ.
- One-minute OHLC cannot reveal TP/SL ordering when both trade inside the same minute.
- Matched controls reduce obvious confounding but do not prove causality.
- FVGs from the same move/day are not fully independent.
- Many timeframe/depth cells were inspected, so post-hoc strong subgroups require independent replication.
- Statistical predictability does not imply net profitability after costs and execution.

## Data licensing

No Databento market data is committed to this repository. The downloader and preparation scripts are provided so readers with their own licensed access can recreate the inputs.

## Portfolio

Full write-up and experiment pages:

**https://t8pium.github.io/projects/fvg-predictive-strength/**
