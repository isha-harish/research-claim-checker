from __future__ import annotations

import json
from pathlib import Path

from checker import load_nli_model, run_nli_check

EXAMPLES_PATH = Path(__file__).with_name("examples.json")


def load_examples():
    with EXAMPLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    tokenizer, model = load_nli_model()
    examples = load_examples()
    total = len(examples)
    correct = 0
    unsupported_as_supported = 0
    failures: list[str] = []

    print(f"Running evaluation on {total} examples...\n")
    for item in examples:
        expected = item["expected_label"]
        predicted = run_nli_check(item["source"], item["claim"], tokenizer=tokenizer, model=model)["result"]
        print(f"{item['id']}: expected={expected} predicted={predicted}")

        if expected == predicted:
            correct += 1
        else:
            failures.append(f"{item['id']}: expected={expected}, predicted={predicted}")

        if expected != "Supported" and predicted == "Supported":
            unsupported_as_supported += 1

    accuracy = correct / total if total else 0.0
    print("\nSummary")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Unsupported claims incorrectly classified as Supported: {unsupported_as_supported}")

    if failures:
        print("\nIndividual failures:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("\nNo failures.")


if __name__ == "__main__":
    main()
