"""AI Research Flywheel — controlled generalization check v2.

Compares the previous selected prompt against the original prompt on a NEW
24-task development set, then evaluates the selected prompt on a separate
12-task validation set that is not used for prompt selection.

No weights or source files are modified. Results are saved separately.

Run from the project directory:
    python research_cycle_v2.py
Requires: benchmark_v2.py, research_cycle_results.json, and local Ollama.
"""
from __future__ import annotations

import json
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import benchmark_v2 as bench

MODEL = bench.MODEL
OLLAMA_URL = bench.OLLAMA_URL
REQUEST_TIMEOUT_SECONDS = 180
MODEL_OPTIONS = {"temperature": 0}
PREVIOUS_REPORT = Path("research_cycle_results.json")
RESULTS_FILE = Path("research_cycle_v2_results.json")

# New tasks: do not reuse M/P/L/R IDs from the original diagnostic benchmark.
NEW_DEVELOPMENT_TESTS: list[dict[str, str]] = [
    # Math
    {"id": "NM01", "category": "Math", "question": "What is 936 divided by 12? Return only the number.", "expected": "78"},
    {"id": "NM02", "category": "Math", "question": "Compute 17 * 19. Return only the number.", "expected": "323"},
    {"id": "NM03", "category": "Math", "question": "Solve 4*x - 11 = 37. Return only the value of x.", "expected": "12"},
    {"id": "NM04", "category": "Math", "question": "Compute 2**5 + 3**3. Return only the number.", "expected": "59"},
    {"id": "NM05", "category": "Math", "question": "Simplify 21/28. Return a simplified fraction only.", "expected": "3/4"},
    {"id": "NM06", "category": "Math", "question": "Find the median of 12, 3, 8, 17, 5. Return only the number.", "expected": "8"},

    # Python semantics and expressions
    {"id": "NP01", "category": "Python", "question": "In Python, what does bool(0) or bool([]) evaluate to? Return True or False only.", "expected": "False"},
    {"id": "NP02", "category": "Python", "question": "In Python, what is the result of 'ha' * 3? Return the resulting string only, without quotes.", "expected": "hahaha"},
    {"id": "NP03", "category": "Python", "question": "In Python, what is sorted([5, 1, 4])[1]? Return only the number.", "expected": "4"},
    {"id": "NP04", "category": "Python", "question": "In Python, what does {'a': 1}.get('missing', 9) return? Return only the number.", "expected": "9"},
    {"id": "NP05", "category": "Python", "question": "In Python, what is len({1, True, 1.0})? Return only the number.", "expected": "1"},
    {"id": "NP06", "category": "Python", "question": "In Python, what does all([True, 1, 'x']) return? Return True or False only.", "expected": "True"},

    # Logic
    {"id": "NL01", "category": "Logic", "question": "Every A is B, and no B is C. Can any A be C under these rules? Answer Yes or No only.", "expected": "No"},
    {"id": "NL02", "category": "Logic", "question": "Every square is a rectangle, and every rectangle is a polygon. Is every square necessarily a polygon? Answer Yes or No only.", "expected": "Yes"},
    {"id": "NL03", "category": "Logic", "question": "Some artists are engineers. All engineers are readers. Must at least one artist be a reader? Answer Yes or No only.", "expected": "Yes"},
    {"id": "NL04", "category": "Logic", "question": "If an alarm sounds, a sensor is triggered. The sensor was not triggered. Could the alarm have sounded under these rules? Answer Yes or No only.", "expected": "No"},
    {"id": "NL05", "category": "Logic", "question": "If P implies Q and Q is true, must P be true? Answer Yes or No only.", "expected": "No"},
    {"id": "NL06", "category": "Logic", "question": "No blue objects are metal. A toy is blue. Can that toy be metal under these rules? Answer Yes or No only.", "expected": "No"},

    # Probability / combinatorics
    {"id": "NR01", "category": "Probability", "question": "Two fair six-sided dice are rolled. What is the probability that their sum is 7? Return a simplified fraction only.", "expected": "1/6"},
    {"id": "NR02", "category": "Probability", "question": "One card is drawn uniformly from a standard 52-card deck. What is the probability it is an ace? Return a simplified fraction only.", "expected": "1/13"},
    {"id": "NR03", "category": "Probability", "question": "A spinner has 8 equal sections, 3 of which are green. What is the probability of landing on green? Return a simplified fraction only.", "expected": "3/8"},
    {"id": "NR04", "category": "Probability", "question": "A bag contains 4 red balls and 6 blue balls. One ball is drawn uniformly at random. What is the probability it is red? Return a simplified fraction only.", "expected": "2/5"},
    {"id": "NR05", "category": "Probability", "question": "A fair coin is tossed three times. What is the probability all three tosses are heads? Return a simplified fraction only.", "expected": "1/8"},
    {"id": "NR06", "category": "Probability", "question": "A fair six-sided die is rolled once. What is the probability of rolling a number less than 3? Return a simplified fraction only.", "expected": "1/3"},
]

# Fresh holdout tasks. Do NOT tune prompt candidates based on these results.
NEW_VALIDATION_TESTS: list[dict[str, str]] = [
    {"id": "XV-M01", "category": "Math", "question": "Solve 9*x + 4 = 85. Return only the value of x.", "expected": "9"},
    {"id": "XV-M02", "category": "Math", "question": "What is 30 percent of 70? Return only the number.", "expected": "21"},
    {"id": "XV-M03", "category": "Math", "question": "Calculate 5/6 - 1/4. Return a simplified fraction only.", "expected": "7/12"},
    {"id": "XV-P01", "category": "Python", "question": "In Python, what does bool('False') evaluate to? Return True or False only.", "expected": "True"},
    {"id": "XV-P02", "category": "Python", "question": "In Python, what does sum([2, 4, 6][1:]) return? Return only the number.", "expected": "10"},
    {"id": "XV-P03", "category": "Python", "question": "In Python, what does 'xy'[::-1] return? Return the resulting string only, without quotes.", "expected": "yx"},
    {"id": "XV-L01", "category": "Logic", "question": "All cats are mammals, and no mammals can fly. Can a cat fly under these rules? Answer Yes or No only.", "expected": "No"},
    {"id": "XV-L02", "category": "Logic", "question": "All A are B, and some B are C. Must some A be C? Answer Yes or No only.", "expected": "No"},
    {"id": "XV-L03", "category": "Logic", "question": "If P implies Q and Q is false, must P be false? Answer Yes or No only.", "expected": "Yes"},
    {"id": "XV-R01", "category": "Probability", "question": "A bag contains 2 red balls and 3 blue balls. What is the probability a randomly drawn ball is not red? Return a simplified fraction only.", "expected": "3/5"},
    {"id": "XV-R02", "category": "Probability", "question": "A fair coin is tossed twice. What is the probability of getting at least one head? Return a simplified fraction only.", "expected": "3/4"},
    {"id": "XV-R03", "category": "Probability", "question": "How many different pairs can be chosen from 6 distinct people? Return only the number.", "expected": "15"},
]


def load_selected_prompt() -> tuple[str, str]:
    if not PREVIOUS_REPORT.exists():
        raise FileNotFoundError(
            f"Missing {PREVIOUS_REPORT}. Run research_cycle.py first and keep it in this folder."
        )
    report = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    prompt = report.get("selected_prompt")
    name = report.get("selected_candidate", "previous_selected_candidate")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"No valid selected_prompt in {PREVIOUS_REPORT}.")
    return str(name), prompt


def ask_model(question: str, system_prompt: str) -> tuple[str, float]:
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
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Make sure Ollama is running. Details: {exc}"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        answer = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response structure: {data!r}") from exc
    return str(answer).strip(), round(elapsed_ms, 1)


def format_check(answer: str, question: str, expected: str) -> bool:
    """Accept only the benchmark checker's known harmless outer math delimiters."""
    clean = bench.clean_model_output(answer)
    clean = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", clean)
    return bench.check_format(clean, question, expected)


def evaluate_set(tests: list[dict[str, str]], label: str, prompt_name: str, prompt: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    print(f"\n=== {label} — prompt: {prompt_name} ({len(tests)} tasks) ===", flush=True)
    for index, task in enumerate(tests, start=1):
        try:
            answer, latency = ask_model(task["question"], prompt)
            correct = bench.check_answer(answer, task["expected"], task["question"])
            fmt = format_check(answer, task["question"], task["expected"])
            error = None
        except Exception as exc:
            answer, latency = "", None
            correct, fmt = False, False
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "id": task["id"], "category": task["category"],
            "question": task["question"], "expected": task["expected"],
            "answer": answer, "correct": bool(correct),
            "format_compliant": bool(fmt), "joint_pass": bool(correct and fmt),
            "latency_ms": latency, "error": error,
        }
        rows.append(row)
        if error:
            status = "ERROR"
        elif correct and fmt:
            status = "PASS"
        elif correct:
            status = "CORRECT / FORMAT FAIL"
        else:
            status = "FAIL"
        latency_text = f"{latency:.0f} ms" if latency is not None else "n/a"
        print(f"[{index}/{len(tests)}] {task['id']} {status} ({latency_text})", flush=True)

    count = len(rows)
    latencies = [float(r["latency_ms"]) for r in rows if r["latency_ms"] is not None]
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "format": 0, "joint": 0})
    for row in rows:
        cat = by_category[row["category"]]
        cat["total"] += 1
        cat["correct"] += int(row["correct"])
        cat["format"] += int(row["format_compliant"])
        cat["joint"] += int(row["joint_pass"])

    summary = {
        "total": count,
        "correct": sum(r["correct"] for r in rows),
        "format_compliant": sum(r["format_compliant"] for r in rows),
        "joint_pass": sum(r["joint_pass"] for r in rows),
        "correct_percent": round(100 * sum(r["correct"] for r in rows) / count, 1) if count else 0.0,
        "format_percent": round(100 * sum(r["format_compliant"] for r in rows) / count, 1) if count else 0.0,
        "joint_percent": round(100 * sum(r["joint_pass"] for r in rows) / count, 1) if count else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "by_category": dict(by_category),
    }
    print(
        f"Summary: correct {summary['correct']}/{count} ({summary['correct_percent']:.1f}%), "
        f"format {summary['format_compliant']}/{count} ({summary['format_percent']:.1f}%), "
        f"joint {summary['joint_pass']}/{count} ({summary['joint_percent']:.1f}%)",
        flush=True,
    )
    for row in rows:
        if not row["joint_pass"]:
            print(f"  ISSUE {row['id']}: expected={row['expected']!r}; answer={row['answer']!r}; "
                  f"correct={row['correct']}; format={row['format_compliant']}", flush=True)
    return {"label": label, "prompt_name": prompt_name, "summary": summary, "details": rows}


def main() -> int:
    selected_name, selected_prompt = load_selected_prompt()
    baseline_prompt = bench.SYSTEM_PROMPT

    print("AI Research Flywheel — controlled generalization check v2")
    print(f"Model: {MODEL}")
    print("Local Ollama only. No model weights or project source files are modified.")
    print(f"Generation options: {MODEL_OPTIONS}")
    print("New development set: baseline vs previously selected prompt.")
    print("Fresh validation set: selected prompt only; do not use it to tune candidates.")
    print(f"Results: {RESULTS_FILE.resolve()}", flush=True)

    baseline_dev = evaluate_set(NEW_DEVELOPMENT_TESTS, "NEW DEVELOPMENT SET", "baseline", baseline_prompt)
    selected_dev = evaluate_set(NEW_DEVELOPMENT_TESTS, "NEW DEVELOPMENT SET", selected_name, selected_prompt)
    selected_validation = evaluate_set(NEW_VALIDATION_TESTS, "FRESH HELD-OUT VALIDATION", selected_name, selected_prompt)

    b = baseline_dev["summary"]
    s = selected_dev["summary"]
    joint_delta = s["joint_pass"] - b["joint_pass"]
    correct_delta = s["correct"] - b["correct"]
    format_delta = s["format_compliant"] - b["format_compliant"]
    if joint_delta > 0 and correct_delta >= 0:
        interpretation = "The selected prompt improves joint pass count on this new development set; repeat with more tasks before promoting it broadly."
    elif joint_delta == 0:
        interpretation = "No joint-pass improvement over baseline on this new development set; check latency and category-level results."
    else:
        interpretation = "The selected prompt performs worse on joint pass count here; do not promote without investigating regressions."

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "generation_options": MODEL_OPTIONS,
        "experiment_type": "controlled prompt generalization evaluation; no weight training",
        "selected_prompt_source": str(PREVIOUS_REPORT),
        "selected_prompt_name": selected_name,
        "selected_prompt": selected_prompt,
        "development_baseline": baseline_dev,
        "development_selected": selected_dev,
        "fresh_validation_selected_only": selected_validation,
        "development_delta_selected_minus_baseline": {
            "correct_count": correct_delta,
            "format_count": format_delta,
            "joint_pass_count": joint_delta,
            "average_latency_ms": (
                round(s["average_latency_ms"] - b["average_latency_ms"], 1)
                if s["average_latency_ms"] is not None and b["average_latency_ms"] is not None else None
            ),
        },
        "interpretation": interpretation,
        "limitations": [
            "The 24-task development set and 12-task validation set are still small diagnostic samples.",
            "A one-run comparison is sensitive to model variability and does not prove broad capability improvement.",
            "The checkers are designed for short deterministic answers, not arbitrary open-ended reasoning.",
            "The selected prompt was chosen in an earlier experiment; this run tests generalization rather than model-weight training.",
        ],
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== CONTROLLED COMPARISON ===")
    print(f"Baseline joint pass: {b['joint_pass']}/{b['total']}")
    print(f"Selected joint pass: {s['joint_pass']}/{s['total']}")
    print(f"Delta: {joint_delta:+d} tasks")
    print(f"Fresh validation joint pass: {selected_validation['summary']['joint_pass']}/{selected_validation['summary']['total']}")
    print(interpretation)
    print(f"\nFull report saved: {RESULTS_FILE.resolve()}")
    print("Existing benchmark and research-cycle reports were not overwritten.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(2)
