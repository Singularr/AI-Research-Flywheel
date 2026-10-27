"""AI Research Flywheel — first controlled prompt-research cycle.

Uses benchmark_v2.py as the source of benchmark tasks and correctness checks.
It compares prompt variants only on selected DEVELOPMENT tasks, selects one
using a fixed rule, then evaluates that selected prompt on the full development
set and the independent validation set. The validation set is not used to
choose the prompt.

This is prompt/configuration research only. It does not train or modify weights.

Run from the project folder:
    python research_cycle.py
"""
from __future__ import annotations

import json
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_v2 as bench

MODEL = bench.MODEL
OLLAMA_URL = bench.OLLAMA_URL
TIMEOUT_SECONDS = 180
MODEL_OPTIONS = {"temperature": 0}
RESULTS_FILE = Path("research_cycle_results.json")

# These are DEVELOPMENT failures/near-failures from the previous run.
# The independent validation set is deliberately not used for prompt selection.
FOCUS_IDS = ["P06", "R01", "R03", "R04"]

BASELINE_PROMPT = bench.SYSTEM_PROMPT

CANDIDATE_PROMPTS: dict[str, str] = {
    "baseline": BASELINE_PROMPT,
    "strict_one_line": (
        "Solve the task accurately. Return exactly one line containing only the final answer. "
        "Never include reasoning, explanation, a heading, a label, Markdown, or LaTeX delimiters. "
        "For a fraction, return a reduced plain-text fraction such as 1/3. "
        "For True/False tasks, return exactly True or False; for Yes/No tasks, return exactly Yes or No. "
        "Before responding, check that the answer itself is correct and that the line follows the requested format."
    ),
    "verified_format": (
        "You are solving a benchmark with strict answer-format requirements. First solve the task and "
        "verify the result internally; do not reveal the verification. Then output exactly the requested "
        "final value on one line. Do not explain, narrate, use a title, or wrap the answer in Markdown or LaTeX. "
        "Use reduced plain-text fractions (for example, 1/3), Python's exact language semantics for code questions, "
        "and the exact Boolean spelling requested by the task. Check your result and output format before answering."
    ),
    "task_type_check": (
        "Solve the problem carefully and silently verify the result. Pay attention to the type of task: "
        "for programming expressions, apply the language's exact runtime semantics, including type-specific truthiness "
        "and Boolean operators; do not guess from the visual appearance of a value. For probability, count equally likely "
        "outcomes and simplify the fraction. "
        "Then output only the requested answer, on exactly one line, without explanation, labels, Markdown, or LaTeX. "
        "Use a reduced plain-text fraction like 1/3 and exactly True/False or Yes/No when requested."
    ),
}


def ask_model_with_prompt(question: str, system_prompt: str) -> tuple[str, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": MODEL_OPTIONS,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Check that Ollama is running. Details: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        answer = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response structure: {data!r}") from exc
    return str(answer).strip(), round(latency_ms, 1)


def strip_math_delimiters(text: str) -> str:
    """Remove only common outer math delimiters before strict format checking."""
    text = bench.clean_model_output(text)
    # Both display-math delimiters must be removed; this fixes a checker false negative.
    text = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", text)
    return text.strip()


def check_format_v4(answer: str, question: str, expected: str) -> bool:
    """Use the benchmark's strict checker after normalizing harmless math wrappers."""
    return bench.check_format(strip_math_delimiters(answer), question, expected)


def evaluate_prompt_on_focus(name: str, prompt: str, tasks: list[dict[str, str]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    print(f"\n--- Candidate: {name} ({len(tasks)} development tasks) ---", flush=True)
    for idx, task in enumerate(tasks, start=1):
        try:
            answer, latency_ms = ask_model_with_prompt(task["question"], prompt)
            correct = bench.check_answer(answer, task["expected"], task["question"])
            format_ok = check_format_v4(answer, task["question"], task["expected"])
            error = None
        except Exception as exc:  # Keep a failed call visible without aborting other trials.
            answer, latency_ms = "", None
            correct, format_ok = False, False
            error = f"{type(exc).__name__}: {exc}"

        details.append({
            "id": task["id"],
            "category": task["category"],
            "question": task["question"],
            "expected": task["expected"],
            "answer": answer,
            "correct": bool(correct),
            "format_compliant": bool(format_ok),
            "latency_ms": latency_ms,
            "error": error,
        })
        if error:
            status = "ERROR"
        elif correct and format_ok:
            status = "PASS"
        elif correct:
            status = "CORRECT / FORMAT FAIL"
        else:
            status = "FAIL"
        latency_text = f"{latency_ms:.0f} ms" if latency_ms is not None else "n/a"
        print(f"[{idx}/{len(tasks)}] {task['id']} {status} ({latency_text})", flush=True)

    n = len(details)
    correct_count = sum(row["correct"] for row in details)
    format_count = sum(row["format_compliant"] for row in details)
    joint_count = sum(row["correct"] and row["format_compliant"] for row in details)
    latencies = [row["latency_ms"] for row in details if row["latency_ms"] is not None]
    return {
        "name": name,
        "prompt": prompt,
        "summary": {
            "total": n,
            "correct": correct_count,
            "format_compliant": format_count,
            "joint_pass": joint_count,
            "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        },
        "details": details,
    }


def evaluate_full_set(tests: list[dict[str, str]], label: str, prompt: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    print(f"\n=== Final evaluation: {label} ({len(tests)} tasks) ===", flush=True)
    for idx, task in enumerate(tests, start=1):
        try:
            answer, latency_ms = ask_model_with_prompt(task["question"], prompt)
            correct = bench.check_answer(answer, task["expected"], task["question"])
            format_ok = check_format_v4(answer, task["question"], task["expected"])
            error = None
        except Exception as exc:
            answer, latency_ms = "", None
            correct, format_ok = False, False
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "id": task["id"],
            "category": task["category"],
            "question": task["question"],
            "expected": task["expected"],
            "answer": answer,
            "correct": bool(correct),
            "format_compliant": bool(format_ok),
            "joint_pass": bool(correct and format_ok),
            "latency_ms": latency_ms,
            "error": error,
        }
        rows.append(row)
        if error:
            status = "ERROR"
        elif correct and format_ok:
            status = "PASS"
        elif correct:
            status = "CORRECT / FORMAT FAIL"
        else:
            status = "FAIL"
        latency_text = f"{latency_ms:.0f} ms" if latency_ms is not None else "n/a"
        print(f"[{idx}/{len(tests)}] {task['id']} {status} ({latency_text})", flush=True)

    n = len(rows)
    correct_count = sum(r["correct"] for r in rows)
    format_count = sum(r["format_compliant"] for r in rows)
    joint_count = sum(r["joint_pass"] for r in rows)
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    categories: dict[str, dict[str, int]] = {}
    for row in rows:
        cat = categories.setdefault(row["category"], {"total": 0, "correct": 0, "format": 0, "joint": 0})
        cat["total"] += 1
        cat["correct"] += int(row["correct"])
        cat["format"] += int(row["format_compliant"])
        cat["joint"] += int(row["joint_pass"])

    summary = {
        "total": n,
        "correct": correct_count,
        "correct_percent": round(100 * correct_count / n, 1) if n else 0.0,
        "format_compliant": format_count,
        "format_percent": round(100 * format_count / n, 1) if n else 0.0,
        "joint_pass": joint_count,
        "joint_percent": round(100 * joint_count / n, 1) if n else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "by_category": categories,
    }
    print(
        f"{label}: correctness {correct_count}/{n} ({summary['correct_percent']:.1f}%), "
        f"format {format_count}/{n} ({summary['format_percent']:.1f}%), "
        f"joint {joint_count}/{n} ({summary['joint_percent']:.1f}%)",
        flush=True,
    )
    for row in rows:
        if not row["correct"] or (row["correct"] and not row["format_compliant"]):
            print(
                f"  {row['id']}: expected={row['expected']!r}; "
                f"correct={row['correct']}; format={row['format_compliant']}; "
                f"answer={row['answer']!r}",
                flush=True,
            )
    return {"label": label, "summary": summary, "details": rows}


def main() -> int:
    print("AI Research Flywheel — controlled prompt experiment v1")
    print(f"Model: {MODEL}")
    print("Local Ollama only; no model weights or source files are modified.")
    print("Selection uses DEVELOPMENT tasks only; independent validation is held out.")
    print(f"Generation options: {MODEL_OPTIONS} (all prompt candidates use the same settings).")
    print(f"Results file: {RESULTS_FILE.resolve()}", flush=True)

    development_by_id = {task["id"]: task for task in bench.DEVELOPMENT_TESTS}
    focus_tasks = [development_by_id[test_id] for test_id in FOCUS_IDS if test_id in development_by_id]
    if len(focus_tasks) != len(FOCUS_IDS):
        missing = sorted(set(FOCUS_IDS) - set(development_by_id))
        raise RuntimeError(f"Focus task IDs are missing from benchmark_v2.py: {missing}")

    candidate_results: list[dict[str, Any]] = []
    for name, prompt in CANDIDATE_PROMPTS.items():
        candidate_results.append(evaluate_prompt_on_focus(name, prompt, focus_tasks))

    # Fixed rule: correctness first, then strict format compliance, then lower latency.
    # We do not score latency ahead of answer quality on this tiny diagnostic subset.
    def ranking_key(candidate: dict[str, Any]) -> tuple[int, int, float]:
        summary = candidate["summary"]
        average_latency = summary["average_latency_ms"]
        return (
            int(summary["correct"]),
            int(summary["format_compliant"]),
            -float(average_latency if average_latency is not None else float("inf")),
        )

    winner = max(candidate_results, key=ranking_key)
    selected_prompt = winner["prompt"]
    print("\n=== Selected candidate ===")
    print(f"Name: {winner['name']}")
    print(f"Development focus results: {winner['summary']}")
    print("Selection rule: highest correctness, then highest format compliance, then lower latency.")
    print("This is a small development-only selection; it is not evidence of general improvement.", flush=True)

    # Now evaluate the selected candidate on all dev tasks and the held-out validation set.
    final_development = evaluate_full_set(bench.DEVELOPMENT_TESTS, "Development set", selected_prompt)
    final_validation = evaluate_full_set(bench.VALIDATION_TESTS, "Independent validation set", selected_prompt)

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "model_options": MODEL_OPTIONS,
        "experiment_type": "prompt selection; no weight training",
        "selection_rule": "correctness descending, format compliance descending, average latency ascending",
        "focus_task_ids": FOCUS_IDS,
        "candidates": [
            {
                "name": candidate["name"],
                "prompt": candidate["prompt"],
                "summary": candidate["summary"],
                "details": candidate["details"],
                "rank_key": list(ranking_key(candidate)),
            }
            for candidate in candidate_results
        ],
        "selected_candidate": winner["name"],
        "selected_prompt": selected_prompt,
        "final_development": final_development,
        "final_validation": final_validation,
        "limitations": [
            "The benchmark is small and the selected prompt may overfit the four focus tasks.",
            "Validation is run only after selection and is not used to choose the prompt.",
            "A single validation run does not prove a general capability gain.",
            "Only prompt behavior is changed; model weights are not trained or modified.",
        ],
    }
    try:
        RESULTS_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"Could not save results: {exc}")
        return 1

    print(f"\nFull report saved: {RESULTS_FILE.resolve()}")
    print("The original benchmark result files were not overwritten.")
    print("Next: inspect whether the selected prompt improves the full development set without harming validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
