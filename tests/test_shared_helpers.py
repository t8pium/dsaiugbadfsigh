import unittest

import numpy as np
import pandas as pd

from fvg_research.bars import market_state, resample_ohlcv
from fvg_research.controls import matched_controls
from fvg_research.stats import chronological_split


class TestSharedHelpers(unittest.TestCase):
    def test_resample_uses_1800_eastern_anchor(self):
        index = pd.date_range("2026-01-01 23:01Z", periods=60, freq="1min")
        frame = pd.DataFrame({
            "open": np.arange(60.), "high": np.arange(60.) + 1,
            "low": np.arange(60.) - 1, "close": np.arange(60.) + .5,
            "volume": 1,
        }, index=index)
        frame.index = frame.index.tz_convert("UTC")
        output = resample_ohlcv(frame, "1H")
        eastern = output.index.tz_convert("America/New_York")
        self.assertTrue(all(eastern.minute == 0))
        self.assertEqual(eastern[0].hour, 19)

    def test_market_state_is_causal(self):
        index = pd.date_range("2026-01-01", periods=300, freq="1min", tz="UTC")
        prices = 100 + np.linspace(0, 2, len(index)) + np.sin(np.arange(len(index)) / 5)
        frame = pd.DataFrame({
            "open": prices, "high": prices + 1, "low": prices - 1,
            "close": prices + .1, "volume": 1,
        }, index=index)
        before = market_state(frame)
        future = frame.iloc[-20:].copy()
        future.index = pd.date_range(index[-1] + pd.Timedelta(minutes=1), periods=20, freq="1min")
        future[["open", "high", "low", "close"]] *= 100
        after = market_state(pd.concat([frame, future])).loc[index]
        pd.testing.assert_series_equal(before.vol_regime, after.vol_regime)

    def test_matching_handles_prejoined_state_and_copies_geometry(self):
        index = pd.date_range("2026-01-01", periods=40, freq="1min", tz="UTC")
        state = pd.DataFrame({
            "open": 100., "close": 100., "atr14": 2., "volatility": .01,
            "session": "ny_am", "tod_30m": 20, "vol_regime": "normal", "trend": 1,
        }, index=index)
        event_time = index[20]
        events = pd.DataFrame({
            "direction": [1], "body_b_atr": [.1], "distance_atr": [.5], "width_atr": [.25],
            "session": ["stale"], "tod_30m": [0], "vol_regime": ["stale"], "trend": [-1],
        }, index=[event_time])
        controls = matched_controls(events, state, n_controls=2)
        self.assertEqual(len(controls), 2)
        self.assertTrue((controls.direction == 1).all())
        self.assertTrue(np.allclose(controls.width, .5))

    def test_chronological_split_validates_fraction(self):
        frame = pd.DataFrame({"x": range(10)}, index=pd.date_range("2026-01-01", periods=10))
        early, late = chronological_split(frame, .7)
        self.assertEqual((len(early), len(late)), (7, 3))
        with self.assertRaises(ValueError):
            chronological_split(frame, 1)


if __name__ == "__main__":
    unittest.main()
