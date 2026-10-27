
import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-r1:8b"

TESTS = [
    {
        "question": "What is 17 + 25? Give the numerical answer.",
        "expected": "42",
    },
    {
        "question": "What Python keyword defines a function?",
        "expected": "def",
    },
    {
        "question": "Which Python type stores key-value pairs? Give its name.",
        "expected": "dict",
    },
]


def ask_deepseek(question):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the question directly and concisely. "
                    "Do not include unrelated information."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["message"]["content"].strip()


def check_answer(answer, expected):
    actual = answer.lower().strip()
    expected = expected.lower().strip()

    aliases = {
        "dict": ["dict", "dictionary"],
    }

    accepted_answers = aliases.get(expected, [expected])

    return any(value in actual for value in accepted_answers)


def main():
    passed = 0
    total = len(TESTS)

    print(f"Model: {MODEL}")
    print(f"Tests: {total}")

    for index, test in enumerate(TESTS, start=1):
        print(f"\nTest {index}: {test['question']}")

        try:
            answer = ask_deepseek(test["question"])
            correct = check_answer(answer, test["expected"])

            print(f"Answer: {answer}")
            print(f"Expected: {test['expected']}")
            print(f"Result: {'PASS' if correct else 'FAIL'}")

            if correct:
                passed += 1

        except Exception as error:
            print(f"Error: {error}")
            print("Check that Ollama is running.")

    print("\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    print(f"Score: {passed / total:.2%}")


if __name__ == "__main__":
    main()
