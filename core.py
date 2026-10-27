
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime, timezone


@dataclass
class ExperimentResult:
    name: str
    score: float
    notes: str = ""


class ResearchFlywheel:
    """Tracks experiment results and compares them with a baseline."""

    def __init__(self, data_file="experiments.json"):
        self.data_file = Path(data_file)
        self.results = self._load_results()

    def _load_results(self):
        if not self.data_file.exists():
            return []

        with self.data_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return [ExperimentResult(**item) for item in data]

    def _save_results(self):
        data = [asdict(result) for result in self.results]
        with self.data_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def record(self, name, score, notes=""):
        if not isinstance(score, (int, float)):
            raise TypeError("Score must be a number.")

        if not name.strip():
            raise ValueError("Experiment name cannot be empty.")

        result = ExperimentResult(name, float(score), notes)
        self.results.append(result)
        self._save_results()
        return result

    def best_result(self):
        if not self.results:
            return None
        return max(self.results, key=lambda result: result.score)

    def report(self):
        if not self.results:
            return "No experiments recorded yet."

        best = self.best_result()
        lines = [f"Experiments recorded: {len(self.results)}"]

        for result in self.results:
            lines.append(f"- {result.name}: {result.score:.4f}")

        lines.append(
            f"Best result: {best.name} ({best.score:.4f})"
        )
        return "\n".join(lines)


if __name__ == "__main__":
    flywheel = ResearchFlywheel()
    print(flywheel.report())
