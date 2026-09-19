import unittest

import numpy as np
import pandas as pd

from fvg_research.fvg import detect_fvgs


def base_frame(periods=30):
    index = pd.date_range("2026-01-01", periods=periods, freq="1min", tz="UTC")
    center = np.full(periods, 100.0)
    return pd.DataFrame({
        "open": center, "high": center + 1, "low": center - 1,
        "close": center + 0.25, "volume": 100,
    }, index=index)


class TestFVGDefinition(unittest.TestCase):
    def test_bullish_and_bearish_geometry(self):
        frame = base_frame(35)
        bull_time = frame.index[20]
        frame.loc[frame.index[18], "high"] = 101
        frame.loc[bull_time, ["open", "high", "low", "close"]] = [102, 104, 102, 103]
        bear_time = frame.index[30]
        frame.loc[frame.index[28], "low"] = 99
        frame.loc[bear_time, ["open", "high", "low", "close"]] = [97, 98, 96, 97]
        events = detect_fvgs(frame)
        bull, bear = events.loc[bull_time], events.loc[bear_time]
        self.assertEqual(int(bull.direction), 1)
        self.assertEqual((bull.lower, bull.upper), (101, 102))
        self.assertEqual(int(bear.direction), -1)
        self.assertEqual((bear.lower, bear.upper), (98, 99))

    def test_no_lookahead_future_mutation(self):
        frame = base_frame(35)
        frame.loc[frame.index[20], ["open", "high", "low", "close"]] = [102, 104, 102, 103]
        before = detect_fvgs(frame).loc[: frame.index[20]]
        frame.loc[frame.index[21]:, ["open", "high", "low", "close"]] += 10_000
        after = detect_fvgs(frame).loc[: frame.index[20]]
        pd.testing.assert_frame_equal(before, after, check_freq=False)

    def test_edge_equality_is_not_gap(self):
        frame = base_frame(25)
        frame.loc[frame.index[20], ["open", "high", "low", "close"]] = [101, 102, 101, 101.5]
        self.assertNotIn(frame.index[20], detect_fvgs(frame).index)

    def test_minimum_tick_is_included_but_subtick_is_not(self):
        exact = base_frame(25)
        exact.loc[exact.index[20], ["open", "high", "low", "close"]] = [101.25, 102, 101.25, 101.5]
        self.assertIn(exact.index[20], detect_fvgs(exact, 0.25).index)
        sub = base_frame(25)
        sub.loc[sub.index[20], ["open", "high", "low", "close"]] = [101.2, 102, 101.2, 101.5]
        self.assertNotIn(sub.index[20], detect_fvgs(sub, 0.25).index)

    def test_invalid_tick_rejected(self):
        with self.assertRaises(ValueError):
            detect_fvgs(base_frame(), 0)


if __name__ == "__main__":
    unittest.main()
