
def score_answer(answer, expected):
    answer_words = answer.lower().split()
    expected_words = expected.lower().split()

    if not expected_words:
        return 0.0

    matches = sum(word in answer_words for word in expected_words)
    return matches / len(expected_words)


def main():
    tests = [
        {
            "name": "Python",
            "expected": "python saves experiment results",
            "baseline": "python saves results",
            "candidate": "python saves experiment results",
        },
        {
            "name": "Math",
            "expected": "2 plus 2 equals 4",
            "baseline": "2 plus 2 equals 5",
            "candidate": "2 plus 2 equals 4",
        },
        {
            "name": "Explanation",
            "expected": "water freezes at zero degrees",
            "baseline": "water gets cold",
            "candidate": "water freezes at zero degrees",
        },
    ]

    baseline_total = 0.0
    candidate_total = 0.0

    for test in tests:
        baseline = score_answer(test["baseline"], test["expected"])
        candidate = score_answer(test["candidate"], test["expected"])

        baseline_total += baseline
        candidate_total += candidate

        print(f"\n{test['name']}")
        print(f"  Baseline:  {baseline:.2%}")
        print(f"  Candidate: {candidate:.2%}")

    count = len(tests)
    baseline_average = baseline_total / count
    candidate_average = candidate_total / count

    print("\n=== FINAL RESULTS ===")
    print(f"Baseline average:  {baseline_average:.2%}")
    print(f"Candidate average: {candidate_average:.2%}")
    print(f"Difference: {(candidate_average - baseline_average):+.2%}")

    if candidate_average > baseline_average:
        print("Decision: candidate performed better on this test suite.")
    elif candidate_average < baseline_average:
        print("Decision: candidate performed worse on this test suite.")
    else:
        print("Decision: no improvement detected.")


if __name__ == "__main__":
    main()
