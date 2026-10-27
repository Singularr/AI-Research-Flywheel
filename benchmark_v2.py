"""AI Research Flywheel — Benchmark v3 evaluator.

Evaluates a local Ollama model on a small, deterministic benchmark.
Reports answer correctness separately from strict output-format compliance.
This script does not train the model or modify model weights.

Run from the project folder:
    python benchmark_v2.py

The report is written to benchmark_v3_results.json so the previous
benchmark_v2_results.json is preserved as the historical baseline.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any


# ---------------------------- Configuration ----------------------------

MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT_SECONDS = 180
RESULTS_FILE = Path("benchmark_v3_results.json")

SYSTEM_PROMPT = (
    "Solve the task carefully, but do not show your reasoning. "
    "Return only the final answer in exactly the format requested by the question. "
    "Do not add explanations, introductions, Markdown, or a boxed answer unless requested."
)


# ------------------------------- Test sets -------------------------------

DEVELOPMENT_TESTS: list[dict[str, str]] = [
    # Mathematics
    {"id": "M01", "category": "Math", "question": "What is 348 + 567? Return only the number.", "expected": "915"},
    {"id": "M02", "category": "Math", "question": "Compute 5 * (18 - 7) + 4. Return only the number.", "expected": "59"},
    {"id": "M03", "category": "Math", "question": "Solve 3*x + 7 = 34. Return only the value of x.", "expected": "9"},
    {"id": "M04", "category": "Math", "question": "What is the remainder when 157 is divided by 12? Return only the remainder.", "expected": "1"},
    {"id": "M05", "category": "Math", "question": "What is the positive square root of 2025? Return only the number.", "expected": "45"},
    {"id": "M06", "category": "Math", "question": "Calculate 3/4 + 2/3. Return a simplified fraction only.", "expected": "17/12"},
    {"id": "M07", "category": "Math", "question": "Compute 2**6 - 3*7. Return only the number.", "expected": "43"},
    {"id": "M08", "category": "Math", "question": "Find the median of 3, 8, 2, 9, 5. Return only the number.", "expected": "5"},

    # Python
    {"id": "P01", "category": "Python", "question": "In Python, what does this print? values = [4, 9, 2, 7]; print(values[1] + values[-1]). Return only the output.", "expected": "16"},
    {"id": "P02", "category": "Python", "question": "What does print(sum(range(1, 6))) output in Python? Return only the output.", "expected": "15"},
    {"id": "P03", "category": "Python", "question": "What does len({'a': 1, 'b': 2, 'c': 3}) return in Python? Return only the number.", "expected": "3"},
    {"id": "P04", "category": "Python", "question": "What does len(set([1, 2, 2, 3, 3, 3])) return in Python? Return only the number.", "expected": "3"},
    {"id": "P05", "category": "Python", "question": "In Python, if s = 'research', what is s[1:4]? Return only the resulting string.", "expected": "ese"},
    {"id": "P06", "category": "Python", "question": "In Python, what does bool([]) or bool('0') evaluate to? Return True or False only.", "expected": "True"},
    {"id": "P07", "category": "Python", "question": "What does this Python code print? numbers = [1, 2, 3]; numbers.append(numbers[0] * 4); print(sum(numbers)). Return only the output.", "expected": "10"},
    {"id": "P08", "category": "Python", "question": "What does print(17 // 4, 17 % 4) output in Python? Return the two numbers separated by one space.", "expected": "4 1"},

    # Logic
    {"id": "L01", "category": "Logic", "question": "All X are Y. All Y are Z. Are all X necessarily Z? Answer Yes or No only.", "expected": "Yes"},
    {"id": "L02", "category": "Logic", "question": "All researchers are readers. Some readers are programmers. Can we conclude that some researchers are programmers? Answer Yes or No only.", "expected": "No"},
    {"id": "L03", "category": "Logic", "question": "If P implies Q, and Q is false, must P be false? Answer Yes or No only.", "expected": "Yes"},
    {"id": "L04", "category": "Logic", "question": "No cats are reptiles. Every tiger is a cat. Can a tiger be a reptile under these rules? Answer Yes or No only.", "expected": "No"},

    # Probability and combinatorics
    {"id": "R01", "category": "Probability", "question": "A fair six-sided die is rolled once. What is the probability of rolling a number greater than 4? Return a simplified fraction only.", "expected": "1/3"},
    {"id": "R02", "category": "Probability", "question": "How many ways can 2 people be chosen from a group of 5? Return only the number.", "expected": "10"},
    {"id": "R03", "category": "Probability", "question": "A bag contains 3 red balls and 2 blue balls. What is the probability of randomly drawing a blue ball? Return a simplified fraction only.", "expected": "2/5"},
    {"id": "R04", "category": "Probability", "question": "A fair coin is tossed twice. What is the probability of getting exactly one head? Return a simplified fraction only.", "expected": "1/2"},
]

# Keep this set separate from prompt development. Do not tune a candidate
# specifically against these questions; use it as a small sanity check only.
VALIDATION_TESTS: list[dict[str, str]] = [
    {"id": "VM01", "category": "Math", "question": "Solve 7*x - 5 = 44. Return only the value of x.", "expected": "7"},
    {"id": "VM02", "category": "Math", "question": "What is 15 percent of 240? Return only the number.", "expected": "36"},
    {"id": "VP01", "category": "Python", "question": "In Python, d = {'x': 4, 'y': 7}; what does print(d['x'] + d.get('z', 3)) output? Return only the output.", "expected": "7"},
    {"id": "VP02", "category": "Python", "question": "In Python, what does print('abc'[-1]) output? Return only the output.", "expected": "c"},
    {"id": "VL01", "category": "Logic", "question": "All engineers use tools. Mira is an engineer. Does Mira use tools under these assumptions? Answer Yes or No only.", "expected": "Yes"},
    {"id": "VL02", "category": "Logic", "question": "Some A are B, and all B are C. Must at least one A be C? Answer Yes or No only.", "expected": "Yes"},
    {"id": "VR01", "category": "Probability", "question": "A bag contains 2 red balls and 3 blue balls. What is the probability of drawing a blue ball? Return a simplified fraction only.", "expected": "3/5"},
    {"id": "VR02", "category": "Probability", "question": "A fair six-sided die is rolled once. What is the probability of an even number? Return a simplified fraction only.", "expected": "1/2"},
]


# ------------------------------ Model call ------------------------------

def ask_model(question: str) -> tuple[str, float]:
    """Ask the local Ollama chat API and return response text and latency ms."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
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
            f"Cannot reach Ollama at {OLLAMA_URL}. Check that Ollama is running. Details: {exc}"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        answer = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response structure: {data!r}") from exc

    return str(answer).strip(), round(elapsed_ms, 1)


# ------------------------------- Evaluators -------------------------------

def clean_model_output(text: str) -> str:
    """Remove explicit reasoning tags and Markdown code fences, if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:\w+)?", "", text, flags=re.I)
    return text.replace("```", "").strip()


def canonicalize_latex(text: str) -> str:
    """Convert common LaTeX fraction/box notation into plain text."""
    text = re.sub(
        r"\\(?:dfrac|tfrac|frac)\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}",
        r"\1/\2",
        text,
    )
    # Repeat to handle a fraction wrapped by \boxed{...} after conversion.
    text = re.sub(r"\\boxed\s*\{\s*([^{}]*)\s*\}", r"\1", text)
    text = re.sub(r"\\text\s*\{\s*([^{}]*)\s*\}", r"\1", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace("$", "")
    return text


def normalize_answer(text: str) -> str:
    """Normalize for answer extraction; does not claim format compliance."""
    text = canonicalize_latex(clean_model_output(text)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Strip only outer punctuation, not arbitrary prose.
    return text.strip(" \t\r\n`*_.,:;!?\"'")


def extract_fraction_candidates(text: str) -> list[Fraction]:
    """Return fractions in order of appearance, ignoring zero denominators."""
    normalized = normalize_answer(text)
    pattern = r"(?<![\w.])([+-]?\d+)\s*/\s*([+-]?\d+)(?![\w.])"
    found: list[Fraction] = []
    for match in re.finditer(pattern, normalized):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator == 0:
            continue
        found.append(Fraction(numerator, denominator))
    return found


def _target_is_fraction(expected: str) -> bool:
    return re.fullmatch(r"\s*[+-]?\d+\s*/\s*[+-]?\d+\s*", expected) is not None


def _target_is_numeric(expected: str) -> bool:
    return re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*", expected) is not None


def _expected_decimal(expected: str) -> Decimal:
    return Decimal(expected.strip())


def check_answer(answer: str, expected: str, question: str = "") -> bool:
    """Check the extracted result separately from output formatting.

    For fractions, evaluate the final fraction in the model output. For a
    question asking specifically for x, prefer the final explicit x assignment
    so a verification equation later in the explanation does not obscure x.
    """
    raw = clean_model_output(answer)
    target = normalize_answer(expected)
    normalized = normalize_answer(raw)

    if normalized == target:
        return True

    if _target_is_fraction(expected):
        candidates = extract_fraction_candidates(raw)
        if not candidates:
            return False
        try:
            wanted = Fraction(expected.strip())
        except (ValueError, ZeroDivisionError):
            return False
        return candidates[-1] == wanted

    if _target_is_numeric(expected):
        try:
            wanted_number = _expected_decimal(expected)
        except InvalidOperation:
            return False

        # If asked for x, the explicit x value is stronger evidence than an
        # arithmetic value appearing later in a verification sentence.
        if re.search(r"\bvalue\s+of\s+x\b|\bsolve\b.*\bx\b", question, re.I):
            assignments = re.findall(
                r"\bx\s*=\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\b",
                raw,
                flags=re.I,
            )
            if assignments:
                try:
                    return Decimal(assignments[-1]) == wanted_number
                except InvalidOperation:
                    return False

        # A boxed final answer, when present, takes priority.
        boxed = re.findall(r"\\boxed\s*\{\s*([^{}]+)\s*\}", raw, flags=re.I)
        for candidate in reversed(boxed):
            candidate = candidate.strip()
            if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", candidate):
                try:
                    return Decimal(candidate) == wanted_number
                except InvalidOperation:
                    pass

        # Otherwise inspect the last standalone number in the response.
        # This is intentionally conservative: arbitrary earlier numbers don't
        # make a response correct if its last numeric result is different.
        numeric_pattern = r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])"
        matches = list(re.finditer(numeric_pattern, canonicalize_latex(raw)))
        if not matches:
            return False
        try:
            return Decimal(matches[-1].group(0)) == wanted_number
        except InvalidOperation:
            return False

    # Yes/No and True/False tasks: judge the final occurrence, not any earlier
    # mention in an explanation.
    if target in {"yes", "no", "true", "false"}:
        tokens = re.findall(r"\b(?:yes|no|true|false)\b", normalized, flags=re.I)
        return bool(tokens) and tokens[-1].lower() == target

    # Text answers: prefer a standalone occurrence bounded by word characters.
    pattern = r"(?<![a-z0-9_])" + re.escape(target) + r"(?![a-z0-9_])"
    return re.search(pattern, normalized, flags=re.I) is not None


def check_format(answer: str, question: str, expected: str) -> bool:
    """Check whether the response is only the requested answer.

    A boxed LaTeX fraction is accepted as answer-only if it contains only that
    fraction. Explanations, labels, and extra lines fail strict compliance.
    """
    raw = clean_model_output(answer)
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(nonempty_lines) != 1:
        return False

    value = nonempty_lines[0].strip()
    # Remove common harmless LaTeX wrappers so \boxed{\dfrac{1}{2}} is read
    # as 1/2, but do not strip explanatory prefixes such as "Answer:".
    value = canonicalize_latex(value).strip()
    value = re.sub(r"\\[,;! ]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("`*_\"'")
    target = normalize_answer(expected)

    # A fraction-only task must contain exactly one reduced fraction.
    if _target_is_fraction(expected):
        match = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*", value)
        if not match:
            return False
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator == 0:
            return False
        fraction = Fraction(numerator, denominator)
        return fraction == Fraction(expected) and fraction.numerator == numerator and fraction.denominator == denominator

    # Boolean / yes-no responses must be a single bare token.
    if target in {"yes", "no", "true", "false"}:
        return value.lower() == target

    # A two-number output such as "4 1" must contain exactly those two numbers.
    if re.fullmatch(r"[-+]?\d+\s+[-+]?\d+", expected.strip()):
        return re.fullmatch(r"[-+]?\d+\s+[-+]?\d+", value) is not None and value == expected.strip()

    # Number-only tasks may return an equivalent integer/decimal representation.
    if _target_is_numeric(expected):
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", value):
            return False
        try:
            return Decimal(value) == Decimal(expected.strip())
        except InvalidOperation:
            return False

    # For strings, quotes around a simple string are harmless; other text is not.
    return value.lower() == target


# ------------------------------- Reporting -------------------------------

def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    correct = sum(bool(row.get("correct", False)) for row in details)
    format_ok = sum(bool(row.get("format_compliant", False)) for row in details)
    latencies = [float(row["latency_ms"]) for row in details if row.get("latency_ms") is not None]

    return {
        "total": total,
        "correct_count": correct,
        "correct_percent": round(100 * correct / total, 1) if total else 0.0,
        "format_count": format_ok,
        "format_percent": round(100 * format_ok / total, 1) if total else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
    }


def evaluate(tests: list[dict[str, str]], label: str) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    print(f"\nEvaluating {label}: {len(tests)} tasks")

    for index, test in enumerate(tests, start=1):
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            answer, latency_ms = ask_model(test["question"])
            correct = check_answer(answer, test["expected"], test["question"])
            format_compliant = check_format(answer, test["question"], test["expected"])
            error = None
        except Exception as exc:  # keep the run/report alive after an individual failure
            answer = ""
            latency_ms = None
            correct = False
            format_compliant = False
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "expected": test["expected"],
            "answer": answer,
            "correct": bool(correct),
            "format_compliant": bool(format_compliant),
            # Keep a passed alias for simple compatibility with older reports.
            "passed": bool(correct),
            "latency_ms": latency_ms,
            "error": error,
            "evaluated_at": started_at,
        }
        details.append(record)

        latency_text = f"{latency_ms:.0f} ms" if latency_ms is not None else "ERROR"
        if error:
            status = "ERROR"
        elif correct and format_compliant:
            status = "PASS"
        elif correct:
            status = "CORRECT / FORMAT FAIL"
        else:
            status = "FAIL"
        print(f"[{index}/{len(tests)}] {test['id']} {status} ({latency_text})")

    return {"label": label, "summary": summarize(details), "details": details}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    details = report["details"]
    print(f"\n=== {report['label']} REPORT ===")
    print(
        f"Answer correctness: {summary['correct_count']}/{summary['total']} "
        f"({summary['correct_percent']:.1f}%)"
    )
    print(
        f"Strict format compliance: {summary['format_count']}/{summary['total']} "
        f"({summary['format_percent']:.1f}%)"
    )

    if summary["average_latency_ms"] is None:
        print("Average response latency: n/a")
        print("Median response latency: n/a")
    else:
        print(f"Average response latency: {summary['average_latency_ms']:.1f} ms")
        print(f"Median response latency: {summary['median_latency_ms']:.1f} ms")

    categories = sorted({row["category"] for row in details})
    print("\nBy category:")
    for category in categories:
        rows = [row for row in details if row["category"] == category]
        correct = sum(bool(row["correct"]) for row in rows)
        format_ok = sum(bool(row["format_compliant"]) for row in rows)
        print(
            f"- {category}: correctness {correct}/{len(rows)} "
            f"({100 * correct / len(rows):.1f}%), format {format_ok}/{len(rows)} "
            f"({100 * format_ok / len(rows):.1f}%)"
        )

    wrong = [row for row in details if not row["correct"]]
    format_failures = [row for row in details if row["correct"] and not row["format_compliant"]]
    errors = [row for row in details if row["error"]]

    print(f"\nIncorrect answers: {len(wrong)}")
    for row in wrong:
        print(f"\n{row['id']} ({row['category']})")
        print(f"Question: {row['question']}")
        print(f"Expected: {row['expected']}")
        print(f"Actual: {row['answer']}")
        if row["error"]:
            print(f"Error: {row['error']}")

    print(f"\nCorrect answer but format failure: {len(format_failures)}")
    for row in format_failures:
        print(f"- {row['id']}: expected answer-only format; actual: {row['answer']}")

    if errors:
        print(f"\nInfrastructure/API errors: {len(errors)}")
        for row in errors:
            print(f"- {row['id']}: {row['error']}")


def main() -> int:
    print("AI Research Flywheel — Benchmark v3 evaluator")
    print(f"Model: {MODEL}")
    print("Local Ollama evaluation; no model weights are modified.")
    print("Correctness and output-format compliance are measured separately.")
    print(f"Results file: {RESULTS_FILE.resolve()}")

    development = evaluate(DEVELOPMENT_TESTS, "Development set")
    validation = evaluate(VALIDATION_TESTS, "Independent validation set")

    print_report(development)
    print_report(validation)

    report = {
        "benchmark_version": "3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "system_prompt": SYSTEM_PROMPT,
        "note": (
            "This benchmark evaluates outputs; it does not train the model. "
            "Correctness is inferred using deterministic answer extraction. "
            "The small task sets are smoke tests, not evidence of general intelligence."
        ),
        "development": development,
        "validation": validation,
    }

    try:
        RESULTS_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"\nCould not save report: {exc}")
        return 1

    print(f"\nResults saved: {RESULTS_FILE.resolve()}")
    print("The previous benchmark_v2_results.json was not overwritten.")
    print("Remember: this is a small diagnostic benchmark; inspect failures before changing prompts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
