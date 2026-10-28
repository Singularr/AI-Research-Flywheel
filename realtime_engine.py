"""Real-time, local Ollama research runner.

Watches research_queue.json for new tasks while running, streams model output,
and appends every event/result to JSONL so interrupted runs remain inspectable.
This is a task-evaluation loop, not model-weight training.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
QUEUE_FILE = Path("research_queue.json")
EVENT_LOG = Path("realtime_events.jsonl")
STATE_FILE = Path("realtime_state.json")
POLL_SECONDS = 1.0
REQUEST_TIMEOUT = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[\s`'\"*_]+|[\s`'\"*_.!,;:]+$", "", value)
    return " ".join(value.split())


def check_answer(answer: str, expected: list[str]) -> bool:
    """Conservative exact-match checker; avoids claiming correctness from substring hits."""
    actual = normalize(answer)
    return any(actual == normalize(item) for item in expected)


class RealtimeResearchEngine:
    def __init__(
        self,
        queue_file: Path = QUEUE_FILE,
        event_log: Path = EVENT_LOG,
        state_file: Path = STATE_FILE,
        model: str = MODEL,
        ollama_url: str = OLLAMA_URL,
        poll_seconds: float = POLL_SECONDS,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        request_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.queue_file = Path(queue_file)
        self.event_log = Path(event_log)
        self.state_file = Path(state_file)
        self.model = model
        self.ollama_url = ollama_url
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.event_callback = event_callback or self._print_event
        self.request_fn = request_fn or self._stream_ollama
        self.stop_requested = False
        self.completed_ids = self._load_completed_ids()
        self.processed_this_run = 0

    @staticmethod
    def _print_event(event: dict[str, Any]) -> None:
        kind = event.get("type", "event")
        if kind == "token":
            print(event.get("text", ""), end="", flush=True)
        else:
            print(f"\n[{kind}] {json.dumps(event, ensure_ascii=False)}", flush=True)

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("timestamp", utc_now())
        self.event_callback(event)
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_state(self, status: str, current_task: str | None = None) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "updated_at": utc_now(),
            "status": status,
            "model": self.model,
            "current_task": current_task,
            "processed_this_run": self.processed_this_run,
            "completed_task_ids": sorted(self.completed_ids),
        }
        temp_path = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_file)

    def _load_completed_ids(self) -> set[str]:
        completed: set[str] = set()
        if not self.event_log.exists():
            return completed
        with self.event_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "result" and event.get("task_id"):
                    completed.add(str(event["task_id"]))
        return completed

    def _read_queue(self) -> list[dict[str, Any]]:
        if not self.queue_file.exists():
            return []
        try:
            data = json.loads(self.queue_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._emit({"type": "queue_error", "error": f"Invalid JSON: {exc}"})
            return []
        if not isinstance(data, list):
            self._emit({"type": "queue_error", "error": "Queue must be a JSON list."})
            return []
        tasks = []
        for index, task in enumerate(data):
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id", "")).strip()
            question = str(task.get("question", "")).strip()
            expected = task.get("expected")
            if not task_id or not question or not isinstance(expected, list) or not expected:
                self._emit({"type": "queue_error", "index": index, "error": "Task requires id, question and non-empty expected list."})
                continue
            if task_id not in self.completed_ids:
                tasks.append({"id": task_id, "question": question, "expected": [str(x) for x in expected]})
        return tasks

    def _stream_ollama(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        parts: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                for raw_line in response:
                    if self.stop_requested:
                        break
                    if not raw_line.strip():
                        continue
                    packet = json.loads(raw_line.decode("utf-8"))
                    message = packet.get("message", {})
                    token = message.get("content", "")
                    if token:
                        parts.append(token)
                        self._emit({"type": "token", "text": token})
                    if packet.get("error"):
                        raise RuntimeError(str(packet["error"]))
                    if packet.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.ollama_url}. Start Ollama and verify model '{self.model}'. Details: {exc}"
            ) from exc
        return "".join(parts).strip()

    def stop(self) -> None:
        """Request a graceful stop after the current streamed response yields control."""
        self.stop_requested = True
        self._emit({"type": "stop_requested"})

    def run_forever(self, max_tasks: int | None = None) -> None:
        self.stop_requested = False
        self._emit({"type": "engine_started", "model": self.model, "queue_file": str(self.queue_file)})
        self._save_state("running")
        try:
            while not self.stop_requested:
                pending = self._read_queue()
                if not pending:
                    self._save_state("waiting_for_tasks")
                    self._emit({"type": "waiting_for_tasks", "message": "Add a new task to research_queue.json, or press Ctrl+C to stop."})
                    time.sleep(self.poll_seconds)
                    continue

                task = pending[0]
                self._save_state("running", task["id"])
                self._emit({"type": "task_started", "task_id": task["id"], "question": task["question"]})
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Answer the task accurately and concisely. Return only the final answer, without unrelated information."},
                        {"role": "user", "content": task["question"]},
                    ],
                    "stream": True,
                    "think": False,
                    "options": {"temperature": 0},
                }
                started = time.perf_counter()
                try:
                    answer = self.request_fn(payload)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    passed = check_answer(answer, task["expected"])
                    result = {
                        "type": "result",
                        "task_id": task["id"],
                        "question": task["question"],
                        "expected": task["expected"],
                        "answer": answer,
                        "passed": passed,
                        "latency_ms": elapsed_ms,
                        "model": self.model,
                        "evaluation": "exact_match",
                    }
                    self.completed_ids.add(task["id"])
                    self.processed_this_run += 1
                    self._emit(result)
                except Exception as exc:
                    self._emit({"type": "task_error", "task_id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
                    # Avoid tight retry loops; the task remains pending and can be retried after delay.
                    time.sleep(self.poll_seconds)
                self._save_state("running")
                if max_tasks is not None and self.processed_this_run >= max_tasks:
                    break
        except KeyboardInterrupt:
            self.stop_requested = True
            self._emit({"type": "keyboard_interrupt"})
        finally:
            self._save_state("stopped")
            self._emit({"type": "engine_stopped", "processed_this_run": self.processed_this_run})


def main() -> int:
    print("AI Research Flywheel — real-time local task runner")
    print(f"Model: {MODEL}")
    print(f"Queue: {QUEUE_FILE.resolve()}")
    print("Streamed answers and events are saved as they arrive.")
    print("Add tasks to research_queue.json while running. Press Ctrl+C to stop.")
    engine = RealtimeResearchEngine()
    engine.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
