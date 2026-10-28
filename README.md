# AI Research Flywheel

## Goal

Build a local AI research system that improves its performance on measurable tasks through controlled experiments.

## Research cycle

1. Define a measurable task.
2. Record baseline performance.
3. Generate a hypothesis for improvement.
4. Run a controlled experiment.
5. Evaluate the result using repeatable tests.
6. Keep changes only when results improve.
7. Record findings and select the next experiment.

## Real-time local runner

The first real-time runner is available in `realtime_engine.py`. It connects to local Ollama, streams answer tokens as they arrive, records events/results to `realtime_events.jsonl`, and atomically updates `realtime_state.json`.

### Requirements

- Python 3.10 or newer.
- Ollama installed and running locally.
- The configured model available locally (default: `deepseek-r1:8b`).

### Start

From the repository folder:

```powershell
ollama list
python -m unittest test_core.py test_realtime_engine.py -v
python realtime_engine.py
```

The runner reads tasks from `research_queue.json`. You can add tasks to that JSON list while the process is running; give every task a unique `id`, a `question`, and a non-empty `expected` list. For example:

```json
{
  "id": "math-new-001",
  "question": "What is 6 * 7? Return only the number.",
  "expected": ["42"]
}
```

Add the object to the existing list, preserving valid JSON. Completed task IDs are read from the event log and are not repeated. Stop the runner with **Ctrl+C**. Results are append-only JSONL events, including failed answers and errors.

### Important limitations

- This first runner continuously executes a user-editable task queue; it does not yet autonomously invent and validate new scientific hypotheses.
- The initial evaluator uses exact-answer matching. It is suitable only for tasks with unambiguous expected answers, not open-ended scientific claims.
- It evaluates model behavior and prompt configuration. It does **not** train or modify model weights.
- A passing score on a small task queue is not evidence of general intelligence or general capability improvement.

## Safety and reliability

- Never claim improvement without measurements.
- Keep a reproducible baseline.
- Run tests before accepting changes.
- Require human approval before modifying core logic.
- Preserve experiment logs and failed results.
- Prefer local data and authorized sources; do not access restricted systems without permission.

## Initial milestone

Create a minimal Python experiment runner with automated tests and a report comparing results against a baseline.
