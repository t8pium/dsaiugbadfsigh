from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fvg_research.dashboard_helpers import (
    discover_results,
    load_latest_success,
    parse_local_paths,
    safe_upload_name,
)

ROOT = Path(__file__).resolve().parent
REF_PATH = ROOT / "reference_results" / "reference_metrics.json"
DATA_FILE = ROOT / "data" / "processed" / "active_mnq.pkl"
RESULTS = ROOT / "results"
UPLOADS = ROOT / "data" / "uploads"
LOGS = RESULTS / "_logs"
UPLOADS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
GITHUB = "https://github.com/t8pium/fvg-predictive-strength"
MAX_BROWSER_UPLOAD = 1024**3
LARGE_UPLOAD_WARNING = 512 * 1024**2

with REF_PATH.open("r", encoding="utf-8") as fh:
    REFERENCE = json.load(fh)

EXPERIMENTS = {
    "raw_fill": {
        "num": "01", "title": "Raw Fill Rates",
        "tagline": "How often do FVGs actually get revisited?",
        "source": "src/original/fvg_final_fast.py", "runner": "detailed",
        "summary": "Measures touch, midpoint mitigation, full fill, and eventual revisit before any control comparison.",
        "method": [
            "Detect each 1-minute FVG only after candle C closes.",
            "Exclude roll-neighborhood events and require valid ATR/state history.",
            "Precompute future rolling extrema for 5, 15, 30, 60, 120, 240, 1,380 and 4,140 one-minute bars.",
            "Test near-edge touch, midpoint mitigation and far-edge/full fill separately.",
            "Use suffix minima/maxima for eventual touch and eventual full fill later in the available dataset.",
            "Interpret the result descriptively only; the next experiment supplies the baseline."
        ],
    },
    "matched_attraction": {
        "num": "02", "title": "Matched-Zone Attraction",
        "tagline": "Are FVGs reached more often than comparable ordinary zones?",
        "source": "src/original/fvg_final_fast.py", "source2": "src/original/fvg_strength_one_tf.py", "runner": "multi",
        "summary": "Falsifies the magnet claim by preserving zone geometry and broad market state while moving the target to non-FVG timestamps.",
        "method": [
            "Sample real FVGs using fixed random seeds.",
            "Use only non-FVG formation bars as control candidates.",
            "Match session, time bucket, volatility regime and trend regime.",
            "Copy the real FVG's direction, ATR-normalized width and ATR-normalized starting distance to each control timestamp.",
            "Measure real and control touches over identical horizons.",
            "Average controls at the parent-FVG level before computing paired differences.",
            "For the deep 1m study, bootstrap paired differences by trading day."
        ],
    },
    "age_decay": {
        "num": "03", "title": "FVG Age Decay",
        "tagline": "Does an old unfilled gap still carry information?",
        "source": "src/original/fvg_strength_one_tf.py", "runner": "multi",
        "summary": "A conditional-survival style test that removes already-filled zones before asking whether the remaining FVGs are still unusually likely to fill.",
        "method": [
            "Reuse real FVGs and their matched ordinary controls.",
            "Compute cumulative touches at 1, 3, 5, 10 and 20 native bars.",
            "For 1→3, keep only zones still untouched after bar 1.",
            "Ask whether those survivors touch by bar 3.",
            "Repeat for 3→5, 5→10 and 10→20.",
            "Compare conditional FVG touch probability against conditional control probability."
        ],
    },
    "continuation": {
        "num": "04", "title": "Formation Continuation",
        "tagline": "Does creating an FVG predict continuation in the same direction?",
        "source": "src/original/fvg_strength_one_tf.py", "runner": "multi",
        "summary": "Matches FVG-forming displacements to similar non-FVG moves and compares subsequent signed returns.",
        "method": [
            "Measure the three-bar displacement and middle-candle body in ATR units.",
            "Classify displacement direction.",
            "Match non-FVG moves by session, volatility regime, trend regime, time bucket, direction, move-size bin and body-size bin.",
            "Sample up to 8,000 FVG moves per timeframe.",
            "Measure signed ATR-normalized return after 1, 3, 5 and 10 native bars.",
            "Compare mean continuation and positive-return frequency."
        ],
    },
    "retest": {
        "num": "05", "title": "First-Touch Retest Reaction",
        "tagline": "Does price react differently once it reaches an FVG?",
        "source": "src/original/fvg_strength_one_tf.py", "runner": "multi",
        "summary": "Treats the FVG as a potential temporary support/resistance region instead of a magnet.",
        "method": [
            "Search up to 20 native bars for the first near-edge touch.",
            "At the touch, set the far edge as full traversal and a rejection target one full gap width away from the near edge.",
            "Begin outcome resolution on the next native bar.",
            "Scan up to 10 bars: rejection first = win, far edge first = loss.",
            "If both boundaries occur in one OHLC bar, mark the race ambiguous.",
            "Also measure normalized close displacement three bars after the touch."
        ],
    },
    "midpoint": {
        "num": "06", "title": "Midpoint / Consequent Encroachment",
        "tagline": "Is the exact 50% level special?",
        "source": "src/original/fvg_midpoint_reaction.py", "runner": "midpoint",
        "summary": "Uses a symmetric race from CE so near and far FVG boundaries are equally distant at the moment the experiment begins.",
        "method": [
            "Resample the active 1m series to the requested native timeframe using the CME 18:00 ET anchor.",
            "Sample up to 12,000 eligible FVGs using seed 9917.",
            "Create three state-matched control zones per FVG.",
            "Search up to 20 native bars for the exact midpoint to trade.",
            "Start the race on the following bar.",
            "Near edge first = rejection success; far edge first = failure; same-bar both = ambiguous.",
            "Bootstrap the paired FVG-control difference by trading day."
        ],
    },
    "body_acceptance": {
        "num": "07", "title": "Candle-Body Acceptance Around CE",
        "tagline": "Does body close depth matter more than the wick?",
        "source": "src/original/fvg_ce_rejection_study.py", "runner": "ce",
        "summary": "Tests opposite-colored candles that wick into the FVG, then classifies their body close as a percentage of gap penetration.",
        "method": [
            "Build 1m through daily bars from the active one-minute series.",
            "Require all three FVG candles to belong to the same listed contract.",
            "For a bullish FVG require a later bearish signal candle; reverse for bearish FVG.",
            "The wick must enter the gap, but wick length itself is ignored.",
            "Normalize body-close depth: 0%=near edge, 50%=CE, 100%=far edge.",
            "Enter at signal close, target the near edge, stop at the far edge.",
            "Resolve higher-timeframe trades on underlying future one-minute bars.",
            "If TP and SL occur in the same one-minute bar, flag the trade ambiguous instead of inventing order."
        ],
    },
    "controls_regimes": {
        "num": "08", "title": "Distance, Regimes & Controlled Model",
        "tagline": "How much does the FVG label matter after obvious variables are controlled?",
        "source": "src/original/fvg_final_fast.py", "runner": "detailed",
        "summary": "Stratifies the 60-minute matched test and fits a logistic model with FVG status alongside distance, size and market-state variables.",
        "method": [
            "Start from the detailed 1m matched FVG/control sample.",
            "Use the 60-minute touch outcome.",
            "Stratify by distance, width, session, direction, volatility, trend, displacement and year.",
            "Repeat under multiple minimum-gap-size filters.",
            "Create a combined FVG/control modeling table.",
            "Fit logistic regression using FVG indicator, distance/ATR, width/ATR, volatility ratio, trend score, direction and cyclical time-of-day.",
            "Report exp(beta_FVG) as the conditional FVG odds ratio."
        ],
    },
    "oos": {
        "num": "09", "title": "Chronological Out-of-Sample Robustness",
        "tagline": "Does the effect survive later data?",
        "source": "src/original/fvg_final_fast.py", "source2": "src/original/fvg_strength_one_tf.py", "runner": "oos",
        "summary": "Uses chronological splits rather than random shuffling to expose non-stationarity and degradation.",
        "method": [
            "Sort matched events chronologically.",
            "Use the first 70% as the early sample and the final 30% as the later sample.",
            "Recompute the 5-native-bar attraction effect separately for each timeframe.",
            "Run a separate deep 1m 60-minute chronological split.",
            "Report matched 60-minute differences by calendar year.",
            "Treat viewed holdouts as research history; future confirmation requires newer or independent data."
        ],
    },
}

def repo_commit() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return "ZIP / no git metadata"

def chart_reference(exp_id: str) -> pd.DataFrame:
    e = REFERENCE["experiments"][exp_id]
    if exp_id == "raw_fill":
        df = pd.DataFrame({"Horizon (minutes/bars)": e["horizons_min"], "Touch probability (%)": [x * 100 for x in e["touch_rate"]]})
        fig = px.line(df, x="Horizon (minutes/bars)", y="Touch probability (%)", markers=True, log_x=True)
        fig.update_layout(title="Published raw 1m FVG touch probability", height=390)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Eventual touch in available sample: {e['eventual_touch']*100:.3f}% · eventual full fill: {e['eventual_full']*100:.3f}%")
        return df
    if exp_id == "matched_attraction":
        df = pd.DataFrame(e["deep_1m"])
        fig = px.bar(df, x="horizon", y="difference_pp", text_auto=".2f")
        fig.update_layout(title="FVG − matched-control touch probability", yaxis_title="Difference (percentage points)", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "age_decay":
        df = pd.DataFrame(e["one_minute"])
        fig = px.bar(df, x="window", y="difference_pp", text_auto=".2f")
        fig.update_layout(title="1m conditional FVG advantage as the gap ages", yaxis_title="Difference (percentage points)", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "continuation":
        df = pd.DataFrame(e["five_bar"])
        fig = px.bar(df, x="timeframe", y="difference_atr", text_auto=".3f")
        fig.update_layout(title="Five-bar continuation: FVG move − matched non-FVG move", yaxis_title="ATR-normalized return difference", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "retest":
        df = pd.DataFrame(e["reaction"])
        fig = px.bar(df, x="timeframe", y="difference_pp", text_auto=".2f")
        fig.update_layout(title="First-touch reaction advantage", yaxis_title="Difference (percentage points)", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "midpoint":
        df = pd.DataFrame(e["reaction"])
        long = df.melt(id_vars=["timeframe"], value_vars=["fvg", "control"], var_name="Zone", value_name="Rejection rate (%)")
        fig = px.bar(long, x="timeframe", y="Rejection rate (%)", color="Zone", barmode="group")
        fig.update_layout(title="Published midpoint rejection race", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "body_acceptance":
        df = pd.DataFrame(e["four_hour_bands"])
        fig = px.bar(df, x="band", y="mean_R", text_auto=".3f", hover_data=["N", "win_rate"])
        fig.update_layout(title="Exploratory 4H body-close bands", yaxis_title="Mean gross R", height=390)
        st.plotly_chart(fig, width="stretch")
        st.warning("The 4H 45–50% cell was discovered after searching multiple bands/timeframes. It is hypothesis-generating, not a confirmed edge.")
        return df
    if exp_id == "controls_regimes":
        df = pd.DataFrame(e["distance"])
        fig = px.line(df, x="bucket", y=["fvg", "control"], markers=True)
        fig.update_layout(title=f"60m touch rate by starting distance · FVG odds ratio ≈ {e['fvg_odds_ratio_60m']:.3f}", yaxis_title="Touch rate (%)", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    if exp_id == "oos":
        df = pd.DataFrame(e["five_bar"])
        long = df.melt(id_vars=["timeframe"], value_vars=["train_pp", "test_pp"], var_name="Split", value_name="Difference (pp)")
        fig = px.bar(long, x="timeframe", y="Difference (pp)", color="Split", barmode="group")
        fig.update_layout(title="Five-native-bar matched attraction: early vs later sample", height=390)
        st.plotly_chart(fig, width="stretch")
        return df
    return pd.DataFrame()

def run_process(args: list[str], env: dict[str, str] | None = None):
    cmd = [str(Path(sys.executable)), *args]
    child_env = os.environ.copy()
    if env:
        child_env.update(env)

    # Child scripts live under scripts/, so make the repository root explicit.
    # This keeps local imports working even if editable-install state is stale.
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")

    log_path = LOGS / f"run_{time.time_ns()}.log"
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, env=child_env,
            )
        content = log_path.read_text(encoding="utf-8", errors="replace")
        if len(content) > 250_000:
            prefix = f"[Earlier output omitted; full log: {log_path.relative_to(ROOT)}]\n"
            content = prefix + content[-250_000:]
        return process.returncode, content
    except OSError as exc:
        return 127, f"Could not start child process: {exc}"

def run_original(study: str, tf: int | None = None, ce_tfs: list[int] | None = None):
    args = [str(ROOT / "scripts" / "run_original.py"), study]
    if tf is not None:
        args += ["--tf", str(tf)]
    if ce_tfs:
        args += ["--ce-tfs", ",".join(map(str, ce_tfs))]
    return run_process(args)


def preparation_args(paths: list[Path]) -> list[str]:
    args = [str(ROOT / "scripts" / "prepare_active_contract.py")]
    for path in paths:
        args.extend(["--input", str(path)])
    return args


def save_uploaded_files(uploaded_files, key_prefix: str) -> list[Path]:
    session_dir = UPLOADS / key_prefix
    session_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for number, uploaded in enumerate(uploaded_files, 1):
        if (getattr(uploaded, "size", 0) or 0) > MAX_BROWSER_UPLOAD:
            raise ValueError(
                f"{uploaded.name} exceeds the 1 GiB browser limit; use the local-path importer."
            )
        target = session_dir / f"{number:02d}_{safe_upload_name(uploaded.name)}"
        uploaded.seek(0)
        with target.open("wb") as dst:
            while True:
                chunk = uploaded.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        uploaded.seek(0)
        paths.append(target)

    return paths


def render_databento_upload(key_prefix: str, compact: bool = False):
    uploaded = st.file_uploader(
        "Upload Databento / OHLCV file(s)",
        type=["dbn", "zst", "parquet", "pq", "csv", "gz", "zip"],
        accept_multiple_files=True,
        key=f"{key_prefix}_market_upload",
        help=(
            "Supported: Databento DBN/DBN.ZST, Parquet, CSV/CSV.GZ, "
            "or ZIP archives containing supported files. You can select multiple files."
        ),
    )

    st.caption(
        "Supported: .dbn, .dbn.zst, .parquet, .pq, .csv, .csv.gz, .zip. "
        "Multiple Databento batch files can be uploaded together."
    )

    if not uploaded:
        return False

    total = sum(getattr(x, "size", 0) or 0 for x in uploaded)
    names = ", ".join(x.name for x in uploaded[:4])
    if len(uploaded) > 4:
        names += f", +{len(uploaded)-4} more"
    st.info(f"Selected {len(uploaded)} file(s) · {total / (1024**2):,.1f} MB\n\n{names}")
    if total > LARGE_UPLOAD_WARNING:
        st.warning(
            "Large browser uploads may duplicate data in memory. For multi-GB data, "
            "use the local-path importer below."
        )

    if st.button(
        "Import file(s) + build active MNQ dataset",
        type="primary",
        key=f"{key_prefix}_process_upload",
        width="stretch" if compact else "content",
    ):
        try:
            saved = save_uploaded_files(uploaded, key_prefix)
        except Exception as exc:
            st.error(f"Could not save uploaded file(s): {exc}")
            return False

        with st.spinner("Reading the upload and rebuilding the active MNQ contract series..."):
            rc, log = run_process(preparation_args(saved))

        st.session_state[f"{key_prefix}_upload_log"] = log
        st.session_state[f"{key_prefix}_upload_rc"] = rc

        if rc == 0 and DATA_FILE.exists():
            st.success("Upload processed successfully. The MNQ dataset is ready for experiments.")
            st.rerun()

        st.error("The file could not be prepared. Open the import log below for details.")

    if st.session_state.get(f"{key_prefix}_upload_log"):
        with st.expander(
            "Import log",
            expanded=st.session_state.get(f"{key_prefix}_upload_rc") != 0,
        ):
            st.code(st.session_state[f"{key_prefix}_upload_log"], language="text")

    return DATA_FILE.exists()


def render_local_path_import(key_prefix: str):
    value = st.text_area(
        "Local file or folder path(s)", key=f"{key_prefix}_local_paths",
        placeholder=r"C:\Users\Test User\Downloads\mnq-batch.zip",
        help="Enter one path per line. Files remain in place and are streamed from disk.",
    )
    st.caption("Recommended for multi-GB datasets. Paths are passed as arguments, never shell commands.")
    if st.button("Import from local path", key=f"{key_prefix}_local_import", disabled=not value.strip()):
        paths = parse_local_paths(value)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            st.error("Path not found: " + ", ".join(missing))
            return False
        with st.spinner("Streaming local data and rebuilding the active MNQ series..."):
            rc, log = run_process(preparation_args(paths))
        st.session_state[f"{key_prefix}_local_log"] = log
        st.session_state[f"{key_prefix}_local_rc"] = rc
        if rc == 0 and DATA_FILE.exists():
            st.success("Local data imported successfully.")
            st.rerun()
        else:
            st.error("Import failed. Review the log below.")
    if st.session_state.get(f"{key_prefix}_local_log"):
        with st.expander("Local import log", expanded=st.session_state.get(f"{key_prefix}_local_rc") != 0):
            st.code(st.session_state[f"{key_prefix}_local_log"], language="text")
    return DATA_FILE.exists()

def result_files() -> list[Path]:
    return discover_results(RESULTS)

def display_result_browser():
    files = result_files()
    if not files:
        st.info("No locally generated result files yet.")
        return
    labels = [str(p.relative_to(ROOT)) for p in files]
    selected = st.selectbox("Generated output", labels, key="output_browser")
    p = ROOT / selected
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        st.dataframe(df, width="stretch", height=min(460, 36 * (len(df) + 1)))
        numeric = list(df.select_dtypes(include="number").columns)
        if len(numeric) >= 2 and len(df) <= 500:
            x = st.selectbox("Chart x", list(df.columns), key="generic_x")
            y = st.selectbox("Chart y", numeric, key="generic_y")
            try:
                st.plotly_chart(px.line(df, x=x, y=y, markers=True), width="stretch")
            except Exception:
                pass
    elif p.suffix == ".json":
        with p.open("r", encoding="utf-8") as fh:
            try:
                st.json(json.load(fh))
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON output: {exc}")
    else:
        st.image(str(p), caption=selected)


def latest_success(study: str) -> dict | None:
    return load_latest_success(RESULTS, DATA_FILE, study)


def fresh_file(path: Path, studies: tuple[str, ...]) -> bool:
    relative = str(path.relative_to(ROOT))
    return any(
        payload and relative in payload.get("generated_files", [])
        for payload in (latest_success(study) for study in studies)
    )


def latest_generated(study: str, prefix: str) -> Path | None:
    payload = latest_success(study)
    if not payload:
        return None
    matches = [ROOT / value for value in payload.get("generated_files", []) if Path(value).name.startswith(prefix)]
    return matches[-1] if matches else None

def verification(exp_id: str):
    rows = []
    try:
        if exp_id == "raw_fill":
            p = RESULTS / "detailed_1m" / "summary.json"
            if fresh_file(p, ("detailed-1m",)):
                actual = json.load(p.open("r", encoding="utf-8"))
                rows += [
                    ("5m touch %", actual["raw_touch_5"] * 100, REFERENCE["experiments"]["raw_fill"]["touch_rate"][0] * 100, 0.35),
                    ("60m touch %", actual["raw_touch_60"] * 100, REFERENCE["experiments"]["raw_fill"]["touch_rate"][3] * 100, 0.35),
                    ("Eventual touch %", actual["eventual_touch_within_dataset"] * 100, REFERENCE["experiments"]["raw_fill"]["eventual_touch"] * 100, 0.35),
                ]
        elif exp_id == "matched_attraction":
            p = RESULTS / "detailed_1m" / "main_results.csv"
            if fresh_file(p, ("detailed-1m",)):
                df = pd.read_csv(p)
                refs = {"touch_5": 3.0275, "touch_60": 0.5770, "touch_1380": 0.0030}
                for test, expected in refs.items():
                    row = df.loc[df["test"] == test]
                    if len(row):
                        rows.append((f"{test} difference pp", float(row.iloc[0]["Difference"]) * 100, expected, 0.50))
            else:
                p = latest_generated("multi-tf", "magnet_tf")
                if p:
                    df = pd.read_csv(p)
                    row = df.loc[df["horizon_bars"] == 5]
                    if len(row):
                        tf = int(row.iloc[0]["timeframe_min"])
                        labels = {1:"1m", 5:"5m", 15:"15m", 60:"1H", 240:"4H"}
                        expected = next(x["difference_pp"] for x in REFERENCE["experiments"]["matched_attraction"]["multi_tf_5bar"] if x["timeframe"] == labels[tf])
                        rows.append((f"{labels[tf]} five-bar difference pp", float(row.iloc[0]["Difference"]) * 100, expected, 0.75))
        elif exp_id == "age_decay":
            p = latest_generated("multi-tf", "decay_tf")
            if p:
                df = pd.read_csv(p)
                row = df[(df["survived_through_bars"] == 1) & (df["next_horizon_bars"] == 3)]
                if len(row) and int(row.iloc[0]["timeframe_min"]) == 1:
                    rows.append(("1→3 bar difference pp", float(row.iloc[0]["Difference"]) * 100, 3.09, 0.75))
        elif exp_id == "continuation":
            p = latest_generated("multi-tf", "formation_tf")
            if p:
                df = pd.read_csv(p)
                row = df[df["horizon_bars"] == 5]
                if len(row):
                    tf = int(row.iloc[0]["timeframe_min"])
                    labels = {1:"1m", 5:"5m", 15:"15m", 60:"1H", 240:"4H"}
                    expected = next(x["difference_atr"] for x in REFERENCE["experiments"]["continuation"]["five_bar"] if x["timeframe"] == labels[tf])
                    rows.append((f"{labels[tf]} five-bar difference ATR", float(row.iloc[0]["Difference_ATR"]), expected, 0.03))
        elif exp_id == "retest":
            p = latest_generated("multi-tf", "reaction_tf")
            if p:
                df = pd.read_csv(p)
                row = df[df["metric"] == "1gap_rejection_before_fullfill"]
                if len(row):
                    tf = int(row.iloc[0]["timeframe_min"])
                    labels = {1:"1m", 5:"5m", 15:"15m", 60:"1H", 240:"4H"}
                    expected = next(x["difference_pp"] for x in REFERENCE["experiments"]["retest"]["reaction"] if x["timeframe"] == labels[tf])
                    rows.append((f"{labels[tf]} reaction difference pp", float(row.iloc[0]["Difference"]) * 100, expected, 1.25))
        elif exp_id == "midpoint":
            p = latest_generated("midpoint", "midpoint_tf")
            if p and "year" not in p.name:
                df = pd.read_csv(p)
                if len(df):
                    tf = int(df.iloc[0]["timeframe_min"])
                    labels = {1:"1m", 5:"5m", 15:"15m", 60:"1H", 240:"4H"}
                    expected = next(x["difference_pp"] for x in REFERENCE["experiments"]["midpoint"]["reaction"] if x["timeframe"] == labels[tf])
                    rows.append((f"{labels[tf]} midpoint difference pp", float(df.iloc[0]["Difference"]) * 100, expected, 1.25))
        elif exp_id == "body_acceptance":
            p = RESULTS / "ce_body" / "body_depth_5pct.csv"
            if fresh_file(p, ("ce-body",)):
                df = pd.read_csv(p)
                row = df[(df["timeframe"] == "4H") & (df["depth_band"] == "45-50%")]
                if len(row):
                    rows.append(("4H 45-50% win rate %", float(row.iloc[0]["win_rate"]) * 100, 67.35, 1.50))
                    rows.append(("4H 45-50% mean R", float(row.iloc[0]["mean_R_conservative"]), 0.286, 0.08))

        elif exp_id == "controls_regimes":
            p = RESULTS / "detailed_1m" / "logistic_60m.json"
            if fresh_file(p, ("detailed-1m",)):
                actual = json.load(p.open("r", encoding="utf-8"))
                rows.append(("FVG odds ratio", float(actual["fvg_odds_ratio"]), 1.0789, 0.03))
        elif exp_id == "oos":
            p = RESULTS / "detailed_1m" / "train_test.csv"
            if fresh_file(p, ("detailed-1m",)):
                df = pd.read_csv(p)
                tr = df[df["split"] == "train"]
                te = df[df["split"] == "test"]
                if len(tr):
                    rows.append(("Deep 1m train difference pp", float(tr.iloc[0]["Difference"]) * 100, 0.79, 0.35))
                if len(te):
                    rows.append(("Deep 1m test difference pp", float(te.iloc[0]["Difference"]) * 100, 0.08, 0.35))
            else:
                p = latest_generated("multi-tf", "oos_tf")
                if p:
                    df = pd.read_csv(p)
                    tf = int(df.iloc[0]["timeframe_min"])
                    labels = {1:"1m", 5:"5m", 15:"15m", 60:"1H", 240:"4H"}
                    reference = next(x for x in REFERENCE["experiments"]["oos"]["five_bar"] if x["timeframe"] == labels[tf])
                    for split, key in (("train", "train_pp"), ("test", "test_pp")):
                        row = df.loc[df["split"] == split]
                        if len(row):
                            rows.append((f"{labels[tf]} {split} difference pp", float(row.iloc[0]["Difference"]) * 100, reference[key], 1.0))
    except Exception as exc:
        st.warning(f"Could not parse verification output: {exc}")
        return
    if not rows:
        st.info("Run the relevant experiment to unlock automatic reference-vs-local checks.")
        return
    out = []
    for name, actual, expected, tolerance in rows:
        delta = actual - expected
        out.append({"Metric": name, "Local run": round(actual, 5), "Published reference": round(expected, 5), "Delta": round(delta, 5), "Tolerance": tolerance, "Status": "PASS" if abs(delta) <= tolerance else "CHECK"})
    st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)

def source_panel(exp):
    files = [exp["source"]] + ([exp["source2"]] if exp.get("source2") else [])
    for rel in files:
        p = ROOT / rel
        st.markdown(f"**{rel}** · [open on GitHub]({GITHUB}/blob/main/{rel})")
        if p.exists():
            with st.expander(f"View {Path(rel).name}"):
                st.code(p.read_text(encoding="utf-8"), language="python")

def reproduce_panel(exp_id: str, exp):
    st.subheader("Reproduce on this machine")
    if not DATA_FILE.exists():
        st.warning("The processed MNQ dataset is not present yet.")
        st.markdown("#### Upload your Databento file here")
        ready = render_databento_upload(f"experiment_{exp_id}", compact=True)
        if ready:
            st.success("Dataset ready. Reopen this Reproduce tab to run the experiment.")
        st.markdown("#### Or use data already on this computer")
        render_local_path_import(f"experiment_{exp_id}")
        with st.expander("Other data setup options"):
            st.write("You can also download the historical data with your own Databento API key.")
            if st.button("Open full Data Setup", key=f"setup_{exp_id}"):
                st.session_state["page"] = "Data Setup"
                st.rerun()
        return
    runner = exp["runner"]
    action = None
    command_preview = ""
    if runner == "detailed":
        command_preview = "python scripts/run_original.py detailed-1m"
        if st.button("Run detailed 1m study", type="primary", key=f"run_{exp_id}"):
            action = ("detailed-1m", None, None)
    elif runner == "multi":
        tf = st.selectbox("Native timeframe", [1, 5, 15, 60, 240], format_func=lambda x: {1:"1m",5:"5m",15:"15m",60:"1H",240:"4H"}[x], key=f"tf_{exp_id}")
        command_preview = f"python scripts/run_original.py multi-tf --tf {tf}"
        if st.button("Run this timeframe", type="primary", key=f"run_{exp_id}"):
            action = ("multi-tf", tf, None)
    elif runner == "midpoint":
        tf = st.selectbox("Native timeframe", [1, 5, 15, 60, 240], format_func=lambda x: {1:"1m",5:"5m",15:"15m",60:"1H",240:"4H"}[x], key=f"tf_{exp_id}")
        command_preview = f"python scripts/run_original.py midpoint --tf {tf}"
        if st.button("Run midpoint experiment", type="primary", key=f"run_{exp_id}"):
            action = ("midpoint", tf, None)
    elif runner == "ce":
        choices = [1,2,3,5,10,15,30,60,120,240,360,480,720,1440]
        selected = st.multiselect("Timeframes to test (minutes)", choices, default=[60,120,240], key=f"cetf_{exp_id}")
        command_preview = "python scripts/run_original.py ce-body --ce-tfs " + ",".join(map(str, selected))
        if st.button("Run body-acceptance study", type="primary", key=f"run_{exp_id}", disabled=not selected):
            action = ("ce-body", None, selected)
    elif runner == "oos":
        mode = st.radio("Validation suite", ["Deep 1m / 60m holdout", "Multi-timeframe / 5-bar holdout"], key="oos_mode")
        if mode.startswith("Deep"):
            command_preview = "python scripts/run_original.py detailed-1m"
            if st.button("Run deep 1m holdout", type="primary", key="run_oos_deep"):
                action = ("detailed-1m", None, None)
        else:
            tf = st.selectbox("Native timeframe", [1,5,15,60,240], format_func=lambda x:{1:"1m",5:"5m",15:"15m",60:"1H",240:"4H"}[x], key="oos_tf")
            command_preview = f"python scripts/run_original.py multi-tf --tf {tf}"
            if st.button("Run timeframe holdout", type="primary", key="run_oos_multi"):
                action = ("multi-tf", tf, None)
    st.code(command_preview, language="bash")
    if action:
        with st.spinner("Running the canonical experiment. Large studies can be CPU/RAM intensive."):
            code, log = run_original(*action)
        st.session_state[f"log_{exp_id}"] = log
        st.session_state[f"rc_{exp_id}"] = code
        if code == 0:
            st.success("Experiment completed. Generated outputs are now available below.")
        else:
            st.error(f"Experiment exited with code {code}.")
    if st.session_state.get(f"log_{exp_id}"):
        with st.expander("Run log", expanded=st.session_state.get(f"rc_{exp_id}") != 0):
            st.code(st.session_state[f"log_{exp_id}"], language="text")

def data_setup():
    st.title("Data Setup")
    st.write("The code is public; the licensed Databento market data is not bundled. This page recreates the input on the reviewer's machine.")
    if DATA_FILE.exists():
        size = DATA_FILE.stat().st_size / (1024**2)
        st.success(f"Processed active-contract dataset found: {DATA_FILE.relative_to(ROOT)} ({size:.1f} MB)")
    else:
        st.warning("Processed dataset not found yet.")
    st.markdown("### Option A — Download with Databento")
    key = st.text_input("Databento API key", type="password", help="Used only for the child download process; the dashboard does not write the key to disk.")
    confirm_cost = st.checkbox("I understand this historical request may incur Databento charges.")
    if st.button("Download + build active MNQ series", type="primary", disabled=not (key and confirm_cost)):
        env = os.environ.copy()
        env["DATABENTO_API_KEY"] = key
        with st.spinner("Downloading licensed data with your Databento account..."):
            rc1, log1 = run_process([str(ROOT/"scripts"/"download_databento.py")], env=env)
        if rc1 == 0:
            with st.spinner("Building the active-contract series..."):
                rc2, log2 = run_process([str(ROOT/"scripts"/"prepare_active_contract.py")])
        else:
            rc2, log2 = 1, ""
        st.code(log1 + "\n" + log2, language="text")
        if rc1 == 0 and rc2 == 0:
            st.success("Dataset prepared. You can now reproduce experiments.")
            st.rerun()
        else:
            st.error("Data preparation failed. Read the log above.")
    st.markdown("### Option B — Upload your own Databento file")
    st.write(
        "Drag the file directly into the app. The Research Lab saves it locally, "
        "reads it, filters MNQ outright contracts, and rebuilds the active one-minute series automatically."
    )
    render_databento_upload("data_setup")

    st.markdown("### Option C — Use files already on this computer")
    st.write("For datasets too large for a browser upload, enter a file or folder path. Nothing is copied into the upload area.")
    render_local_path_import("data_setup")

    with st.expander("Manual file-path fallback"):
        st.write("If you prefer not to use the uploader, the command-line importer also accepts one or more files:")
        st.code(
            "python scripts/prepare_active_contract.py --input YOUR_FILE.dbn.zst\n"
            "python scripts/prepare_active_contract.py --input PART1.dbn.zst --input PART2.dbn.zst",
            language="bash",
        )
    st.markdown("### Code self-test")
    st.write("This deterministic synthetic test requires no market-data license. It checks that the core detector finds known bullish/bearish FVGs and only creates the event on candle C.")
    if st.button("Run synthetic detector self-test"):
        rc, log = run_process(["-m", "unittest", "discover", "-s", "tests", "-v"])
        st.code(log, language="text")
        if rc == 0:
            st.success("Synthetic detector self-test passed.")
        else:
            st.error("Self-test failed. Inspect the log before trusting local reproduction.")

    st.markdown("### Expected published snapshot")
    st.code("active rows: 2,303,483\ncontracts: 27\nstart: 2020-01-01 23:00:00+00:00\nend: 2026-07-10 20:59:00+00:00\nduplicate timestamps: 0\nmissing OHLC: 0", language="text")
    st.caption("If vendor history has been corrected since the published run, tiny numerical differences can be legitimate.")

def home():
    st.markdown("# FVG Predictive Strength — Research Lab")
    st.markdown("A local, inspectable interface for the MNQ Fair Value Gap study. **Published** values below are frozen reference results. **Local run** values come only from code executed on your machine.")
    a,b,c,d = st.columns(4)
    a.metric("Base bars", f"{REFERENCE['study']['active_1m_rows']:,}")
    b.metric("1m FVGs", f"{REFERENCE['study']['fvg_1m_count']:,}")
    c.metric("Contracts", REFERENCE["study"]["contracts"])
    d.metric("Code version", repo_commit())
    st.info("Start with any experiment card. To independently reproduce the numbers, use Data Setup and then the Run section inside that experiment.")
    items = list(EXPERIMENTS.items())
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (exp_id, exp) in zip(cols, items[row_start:row_start+3]):
            with col:
                with st.container(border=True):
                    st.caption(f"EXPERIMENT {exp['num']}")
                    st.markdown(f"### {exp['title']}")
                    st.write(exp["tagline"])
                    st.caption(exp["summary"])
                    if st.button("Open experiment →", key=f"open_{exp_id}", width="stretch"):
                        st.session_state["selected"] = exp_id
                        st.session_state["page"] = "Experiment"
                        st.rerun()
    st.markdown("---")
    st.markdown(f"[Open source repository]({GITHUB}) · [Download repository ZIP]({GITHUB}/archive/refs/heads/main.zip) · [Portfolio report](https://t8pium.github.io/projects/fvg-predictive-strength/)")

def experiment_page(exp_id: str):
    exp = EXPERIMENTS[exp_id]
    if st.button("← All experiments"):
        st.session_state["page"] = "Home"
        st.rerun()
    st.caption(f"EXPERIMENT {exp['num']}")
    st.title(exp["title"])
    st.subheader(exp["tagline"])
    st.write(exp["summary"])
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Method", "Published evidence", "Source code", "Reproduce", "Local outputs"])
    with tab1:
        st.markdown("### Exact procedure")
        for i, step in enumerate(exp["method"], 1):
            st.markdown(f"**{i}.** {step}")
        st.markdown(f"[Full technical documentation]({GITHUB}/blob/main/docs/EXPERIMENTS.md)")
    with tab2:
        st.caption("FROZEN PUBLISHED RESULT — loaded from reference_results/reference_metrics.json")
        df = chart_reference(exp_id)
        if len(df):
            st.dataframe(df, width="stretch", hide_index=True)
    with tab3:
        st.caption("CANONICAL ANALYSIS SOURCE")
        source_panel(exp)
    with tab4:
        reproduce_panel(exp_id, exp)
        st.markdown("### Automatic reference check")
        verification(exp_id)
    with tab5:
        st.write("These are files generated locally by the experiment scripts in this extracted copy.")
        display_result_browser()

st.set_page_config(page_title="FVG Predictive Strength — Research Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2rem; padding-bottom: 4rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: rgba(18,22,27,.45);}
code {font-size: .85em;}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

with st.sidebar:
    st.markdown("## FVG Research Lab")
    if st.button("Home", width="stretch"):
        st.session_state["page"] = "Home"; st.rerun()
    if st.button("Data Setup", width="stretch"):
        st.session_state["page"] = "Data Setup"; st.rerun()
    if st.button("Generated Outputs", width="stretch"):
        st.session_state["page"] = "Generated Outputs"; st.rerun()
    st.markdown("---")
    st.caption("Dataset")
    st.write("✅ ready" if DATA_FILE.exists() else "⚠️ not prepared")
    st.caption("Published study")
    st.write("MNQ · 2020–2026")
    st.markdown(f"[GitHub source ↗]({GITHUB})")

if st.session_state["page"] == "Data Setup":
    data_setup()
elif st.session_state["page"] == "Generated Outputs":
    st.title("Generated Outputs")
    st.write("Browse result files created by experiments on this machine.")
    display_result_browser()
elif st.session_state["page"] == "Experiment" and st.session_state.get("selected") in EXPERIMENTS:
    experiment_page(st.session_state["selected"])
else:
    home()
