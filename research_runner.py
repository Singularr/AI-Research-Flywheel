
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ollama_experiment import TESTS, ask_deepseek, check_answer


RESULTS_FILE = Path("research_results.json")

PROMPTS = [
    {
        "name": "baseline",
        "system": (
            "Answer the question directly and concisely. "
            "Do not include unrelated information."
        ),
    },
    {
        "name": "structured",
        "system": (
            "Answer accurately and concisely. "
            "Check your answer before responding. "
            "Return only the information needed to answer the question."
        ),
    },
]


def ask_with_prompt(question, system_prompt):
    payload = {
        "model": "deepseek-r1:8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["message"]["content"].strip()


def main():
    results = []

    for prompt in PROMPTS:
        passed = 0
        print(f"\nTesting variant: {prompt['name']}")

        for test in TESTS:
            try:
                answer = ask_with_prompt(
                    test["question"],
                    prompt["system"],
                )
                correct = check_answer(answer, test["expected"])

                if correct:
                    passed += 1

                print(
                    f"{test['question']} "
                    f"=> {'PASS' if correct else 'FAIL'}"
                )

            except Exception as error:
                print(f"Test error: {error}")

        score = passed / len(TESTS)

        results.append({
            "name": prompt["name"],
            "passed": passed,
            "total": len(TESTS),
            "score": score,
        })

        print(f"Score: {passed}/{len(TESTS)} ({score:.2%})")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-r1:8b",
        "results": results,
    }

    history = []

    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)

    history.append(record)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    print("\n=== COMPARISON ===")

    for result in results:
        print(f"{result['name']}: {result['score']:.2%}")

    best = max(results, key=lambda item: item["score"])
    print(f"Highest score this run: {best['name']}")
    print(f"Saved experiment runs: {len(history)}")
    print(f"Results file: {RESULTS_FILE.resolve()}")


if __name__ == "__main__":
    main()
