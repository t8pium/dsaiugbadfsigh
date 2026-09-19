# Local data area

Licensed MNQ data is not redistributed. `raw/`, `uploads/`, and `processed/` are ignored by Git.

Use the dashboard to upload a supported Databento/OHLCV file, stream a local path, or download with your own API key. The preparation step is `scripts/prepare_active_contract.py` (there is no `prepare_data.py`).

The published snapshot contains 2,303,483 active one-minute rows, 27 contracts, no duplicate timestamps, and no missing OHLC fields from 2020-01-01 through 2026-07-10.
