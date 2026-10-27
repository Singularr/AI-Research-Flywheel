
# AI Research Flywheel

## Goal

Build a local AI research system that improves its performance on
measurable tasks through controlled experiments.

## Research cycle

1. Define a measurable task.
2. Record baseline performance.
3. Generate a hypothesis for improvement.
4. Run a controlled experiment.
5. Evaluate the result using repeatable tests.
6. Keep changes only when results improve.
7. Record findings and select the next experiment.

## Safety and reliability

- Never claim improvement without measurements.
- Keep a reproducible baseline.
- Run tests before accepting changes.
- Require human approval before modifying core logic.
- Preserve experiment logs and failed results.

## Initial milestone

Create a minimal Python experiment runner with automated tests
and a report comparing results against a baseline.
