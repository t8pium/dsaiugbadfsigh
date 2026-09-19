from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
UPLOADS = DATA / "uploads"
RESULTS = ROOT / "results"

ACTIVE_1M = PROCESSED / "mnq_active_1m.parquet"
ACTIVE_PICKLE = PROCESSED / "active_mnq.pkl"
FVG_EVENTS_1M = PROCESSED / "fvg_events_1m.parquet"

SEED = 20260918
TICK_SIZE = 0.25
START = "2020-01-01"
# Databento range ends are exclusive. This includes all of 2026-07-10.
END = "2026-07-11"

for path in (RAW, PROCESSED, UPLOADS, RESULTS):
    path.mkdir(parents=True, exist_ok=True)
