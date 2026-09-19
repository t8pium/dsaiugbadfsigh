import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class TestDashboardRoutes(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=30)

    def assert_clean(self, app):
        self.assertEqual(list(app.exception), [], [x.value for x in app.exception])

    def test_home_data_and_outputs(self):
        app = self.app().run()
        self.assert_clean(app)
        self.assertTrue(any("Research Lab" in block.value for block in app.markdown))
        for page in ("Data Setup", "Generated Outputs"):
            app.session_state["page"] = page
            app.run()
            self.assert_clean(app)

    def test_all_nine_experiment_routes_without_market_data(self):
        experiment_ids = [
            "raw_fill", "matched_attraction", "age_decay", "continuation", "retest",
            "midpoint", "body_acceptance", "controls_regimes", "oos",
        ]
        for experiment_id in experiment_ids:
            with self.subTest(experiment=experiment_id):
                app = self.app().run()
                app.session_state["page"] = "Experiment"
                app.session_state["selected"] = experiment_id
                app.run()
                self.assert_clean(app)
                self.assertGreaterEqual(len(app.tabs), 5)


if __name__ == "__main__":
    unittest.main()
