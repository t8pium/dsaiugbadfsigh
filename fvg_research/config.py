from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

ACTIVE_1M = PROCESSED / "mnq_active_1m.parquet"
FVG_EVENTS_1M = PROCESSED / "fvg_events_1m.parquet"

SEED = 20260918
TICK_SIZE = 0.25
START = "2020-01-01"
END = "2026-07-10"

for p in (RAW, PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
