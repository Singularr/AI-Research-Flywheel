import json
import tempfile
import unittest
from pathlib import Path

from realtime_engine import RealtimeResearchEngine, check_answer


class TestRealtimeResearchEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.queue = root / "research_queue.json"
        self.log = root / "events.jsonl"
        self.state = root / "state.json"
        self.queue.write_text(json.dumps([
            {"id": "task-1", "question": "What is 2 + 2?", "expected": ["4"]}
        ]), encoding="utf-8")
        self.events = []

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_answer_checker(self):
        self.assertTrue(check_answer(" 4 ", ["4"]))
        self.assertFalse(check_answer("The answer is 4", ["4"]))
        self.assertFalse(check_answer("14", ["4"]))

    def test_task_runs_and_persists_result(self):
        engine = RealtimeResearchEngine(
            queue_file=self.queue,
            event_log=self.log,
            state_file=self.state,
            poll_seconds=0.1,
            event_callback=self.events.append,
            request_fn=lambda payload: "4",
        )
        engine.run_forever(max_tasks=1)
        records = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        result = next(row for row in records if row["type"] == "result")
        self.assertTrue(result["passed"])
        self.assertEqual(result["task_id"], "task-1")
        self.assertTrue(self.state.exists())
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))["status"], "stopped")

    def test_completed_task_is_not_repeated(self):
        self.log.write_text(json.dumps({"type": "result", "task_id": "task-1"}) + "\n", encoding="utf-8")
        engine = RealtimeResearchEngine(
            queue_file=self.queue,
            event_log=self.log,
            state_file=self.state,
            poll_seconds=0.1,
            event_callback=self.events.append,
            request_fn=lambda payload: self.fail("completed task must not rerun"),
        )
        # Empty queue waits until stop is requested; use a callback to stop after waiting.
        def callback(event):
            self.events.append(event)
            if event["type"] == "waiting_for_tasks":
                engine.stop()
        engine.event_callback = callback
        engine.run_forever()
        self.assertFalse(any(event.get("type") == "result" for event in self.events))

    def test_streamed_payload_enabled(self):
        captured = {}
        def fake(payload):
            captured.update(payload)
            return "4"
        engine = RealtimeResearchEngine(
            queue_file=self.queue,
            event_log=self.log,
            state_file=self.state,
            poll_seconds=0.1,
            event_callback=self.events.append,
            request_fn=fake,
        )
        engine.run_forever(max_tasks=1)
        self.assertTrue(captured["stream"])
        self.assertEqual(captured["model"], "deepseek-r1:8b")


if __name__ == "__main__":
    unittest.main()
