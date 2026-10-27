

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_FILE = Path("auto_research_results.json")
MAX_EXPERIMENTS = 3

# Задачи для выбора кандидатов.
TRAIN_TESTS = [
    {"question": "What is 17 + 25? Reply with the number only.",
     "expected": ["42"]},
    {"question": "What is 9 multiplied by 8? Reply with the number only.",
     "expected": ["72"]},
    {"question": "What Python keyword defines a function?",
     "expected": ["def"]},
    {"question": "Which Python type stores key-value pairs?",
     "expected": ["dict", "dictionary"]},
    {"question": "What does this code print? Reply with the number only:\nx = 3\nx = x + 2\nprint(x)",
     "expected": ["5"]},
    {"question": "Which Python function returns the length of a list?",
     "expected": ["len"]},
]

# Отдельные контрольные задачи.
VALIDATION_TESTS = [
    {"question": "What is 15 // 4 in Python? Reply with the number only.",
     "expected": ["3"]},
    {"question": "What does bool([]) return in Python? Reply True or False.",
     "expected": ["false"]},
    {"question": "Which list method adds one item to the end of a Python list?",
     "expected": ["append"]},
    {"question": "What does this code print? Reply with the number only:\nprint(2 ** 5)",
     "expected": ["32"]},
]


def ask_model(messages, temperature=0):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["message"]["content"].strip()


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"^[\s`'\"*_]+|[\s`'\"*_.!,;:]+$", "", text)
    return " ".join(text.split())


def check_answer(answer, expected):
    actual = normalize(answer)
    accepted = [normalize(item) for item in expected]

    # Сначала проверяем точное совпадение.
    if actual in accepted:
        return True

    # Для слов учитываем, что модель может добавить скобки
    # или объяснение к правильному ответу.
    for item in accepted:
        # Числа проверяем только точным совпадением,
        # чтобы не принять 142 за 42.
        if item.isdigit():
            continue

        pattern = rf"(?<![a-z0-9_]){re.escape(item)}(?![a-z0-9_])"

        if re.search(pattern, actual):
            return True

    return False


def evaluate(system_prompt, tests):
    passed = 0
    details = []

    for test in tests:
        try:
            answer = ask_model([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["question"]},
            ])

            correct = check_answer(answer, test["expected"])
            error = None

        except Exception as exc:
            answer = ""
            correct = False
            error = str(exc)

        details.append({
            "question": test["question"],
            "expected": test["expected"],
            "answer": answer,
            "passed": correct,
            "error": error,
        })
        passed += int(correct)

    return {
        "score": passed / len(tests),
        "passed": passed,
        "total": len(tests),
        "details": details,
    }


def print_score(label, result):
    print(
        f"{label}: {result['passed']}/{result['total']} "
        f"({result['score']:.0%})"
    )


def main():
    print("AI Research Flywheel — controlled prompt experiments")
    print(f"Model: {MODEL}")
    print(f"Selection tests: {len(TRAIN_TESTS)}")
    print(f"Validation tests: {len(VALIDATION_TESTS)}")
    print(f"Candidate experiments: {MAX_EXPERIMENTS}")

    baseline_prompt = (
        "Answer the question directly and concisely. "
        "Do not include unrelated information."
    )

    print("\nEvaluating baseline...")
    baseline_train = evaluate(baseline_prompt, TRAIN_TESTS)
    baseline_validation = evaluate(
        baseline_prompt, VALIDATION_TESTS
    )

    print_score("Baseline selection score", baseline_train)
    print_score("Baseline validation score", baseline_validation)

    best_prompt = baseline_prompt
    best_train = baseline_train
    best_validation = baseline_validation

    history = [{
        "kind": "baseline",
        "prompt": baseline_prompt,
        "selection": baseline_train,
        "validation": baseline_validation,
        "accepted": True,
    }]

    for number in range(1, MAX_EXPERIMENTS + 1):
        print(f"\n--- Experiment {number} ---")

        try:
            proposal = ask_model([
                {
                    "role": "system",
                    "content": (
                        "You design system prompts for accurate answers "
                        "to Python and math questions. Return only one "
                        "complete proposed system prompt. Do not include "
                        "explanations, scores, or markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current best prompt:\n{best_prompt}\n\n"
                        f"Selection score: {best_train['score']:.0%}\n"
                        "Propose a meaningfully different prompt to test. "
                        "Do not claim it is better before evaluation."
                    ),
                },
            ], temperature=0.7)

            if not proposal or len(proposal) > 3000:
                raise ValueError("Invalid or excessively long prompt.")

            candidate_train = evaluate(proposal, TRAIN_TESTS)
            print_score("Candidate selection score", candidate_train)

            accepted = False
            candidate_validation = None

            # Only validate candidates that improve the selection score.
            if candidate_train["score"] > best_train["score"]:
                candidate_validation = evaluate(
                    proposal, VALIDATION_TESTS
                )
                print_score(
                    "Candidate validation score", candidate_validation
                )

                # Require improvement on both sets.
                if (
                    candidate_validation["score"]
                    > best_validation["score"]
                ):
                    best_prompt = proposal
                    best_train = candidate_train
                    best_validation = candidate_validation
                    accepted = True
                    print("Decision: candidate provisionally accepted.")
                else:
                    print(
                        "Decision: rejected; validation improvement "
                        "was not demonstrated."
                    )
            else:
                print(
                    "Decision: rejected; selection score did not improve."
                )

            history.append({
                "kind": "candidate",
                "experiment": number,
                "prompt": proposal,
                "selection": candidate_train,
                "validation": candidate_validation,
                "accepted": accepted,
            })

        except Exception as exc:
            print(f"Experiment error: {exc}")
            history.append({
                "kind": "error",
                "experiment": number,
                "error": str(exc),
                "accepted": False,
            })

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "baseline_selection_score": baseline_train["score"],
        "baseline_validation_score": baseline_validation["score"],
        "best_selection_score": best_train["score"],
        "best_validation_score": best_validation["score"],
        "best_prompt": best_prompt,
        "experiments": history,
    }

    previous = []
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as file:
            previous = json.load(file)

        if not isinstance(previous, list):
            raise ValueError(
                "Results file must contain a JSON list. "
                "No previous data was overwritten."
            )

    previous.append(record)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(previous, file, ensure_ascii=False, indent=2)

    print("\n=== FINAL REPORT ===")
    print_score("Baseline selection", baseline_train)
    print_score("Best selection", best_train)
    print_score("Baseline validation", baseline_validation)
    print_score("Best validation", best_validation)
    print(f"Runs saved: {len(previous)}")
    print(f"Report: {RESULTS_FILE.resolve()}")
    print("\nNo model weights or project source files were changed.")
    print("Scores are preliminary; the test set is still small.")


if __name__ == "__main__":
    main()
