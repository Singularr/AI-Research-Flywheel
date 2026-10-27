
import json
import tempfile
import unittest
from pathlib import Path

from core import ResearchFlywheel


class TestResearchFlywheel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_file = Path(self.temp_dir.name) / "experiments.json"
        self.flywheel = ResearchFlywheel(str(self.data_file))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_flywheel(self):
        self.assertIsNone(self.flywheel.best_result())
        self.assertEqual(
            self.flywheel.report(),
            "No experiments recorded yet."
        )

    def test_record_experiment(self):
        result = self.flywheel.record(
            "baseline", 0.5, "Initial measurement"
        )
        self.assertEqual(result.name, "baseline")
        self.assertEqual(result.score, 0.5)
        self.assertTrue(self.data_file.exists())

    def test_best_result(self):
        self.flywheel.record("baseline", 0.5)
        self.flywheel.record("improved", 0.8)
        self.assertEqual(
            self.flywheel.best_result().name,
            "improved"
        )

    def test_results_survive_reload(self):
        self.flywheel.record("baseline", 0.5)
        reloaded = ResearchFlywheel(str(self.data_file))
        self.assertEqual(len(reloaded.results), 1)
        self.assertEqual(reloaded.results[0].score, 0.5)

    def test_empty_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.flywheel.record("", 0.5)

    def test_nonnumeric_score_is_rejected(self):
        with self.assertRaises(TypeError):
            self.flywheel.record("invalid", "high")

    def test_json_is_valid(self):
        self.flywheel.record("baseline", 0.5)
        with self.data_file.open(encoding="utf-8") as file:
            data = json.load(file)
        self.assertEqual(data[0]["name"], "baseline")


if __name__ == "__main__":
    unittest.main()
