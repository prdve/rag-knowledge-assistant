import sys
from pathlib import Path
import csv
import time

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_pipeline import rag_answer


TEST_CASES = [

    {
        "id": "R01",
        "question":
        "What is retrieval augmented generation?",
        "should_answer": True
    },

    {
        "id": "R02",
        "question":
        "What are embeddings?",
        "should_answer": True
    },

    {
        "id": "R03",
        "question":
        "Why are vector databases used in RAG?",
        "should_answer": True
    },

    {
        "id": "R04",
        "question":
        "What is prompt engineering?",
        "should_answer": True
    },

    {
        "id": "R05",
        "question":
        "What is chunking in a RAG system?",
        "should_answer": True
    },

    {
        "id": "R06",
        "question":
        "How does retrieval work?",
        "should_answer": True
    },

    {
        "id": "R07",
        "question":
        "Who won the FIFA World Cup in 1970?",
        "should_answer": False
    },

    {
        "id": "R08",
        "question":
        "What is the population of Brazil today?",
        "should_answer": False
    },

    {
        "id": "R09",
        "question":
        "Who is the CEO of Tesla?",
        "should_answer": False
    },

    {
        "id": "R10",
        "question":
        "What is the recipe for chocolate cake?",
        "should_answer": False
    }

]


ABSTENTION_TEXT = (
    "i don't have enough information "
    "in the knowledge base"
)


def evaluate_case(
    test,
    top_k
):

    start = time.perf_counter()

    try:

        result = rag_answer(
            test["question"],
            top_k=top_k
        )

        latency = (
            time.perf_counter()
            - start
        )

        answer = result[
            "answer"
        ]

        abstained = (
            ABSTENTION_TEXT
            in answer.lower()
        )

        if test["should_answer"]:

            behavior_correct = (
                not abstained
            )

        else:

            behavior_correct = (
                abstained
            )

        return {

            "id":
            test["id"],

            "question":
            test["question"],

            "should_answer":
            test["should_answer"],

            "top_k":
            top_k,

            "abstained":
            abstained,

            "behavior_correct":
            behavior_correct,

            "latency_seconds":
            round(latency, 2),

            "answer":
            answer

        }

    except Exception as e:

        return {

            "id":
            test["id"],

            "question":
            test["question"],

            "should_answer":
            test["should_answer"],

            "top_k":
            top_k,

            "abstained":
            False,

            "behavior_correct":
            False,

            "latency_seconds":
            0,

            "answer":
            f"ERROR: {e}"

        }


def main():

    all_results = []

    for top_k in [1, 3, 5]:

        print(
            f"\nTesting TOP_K = {top_k}"
        )

        print("=" * 60)

        for test in TEST_CASES:

            print(
                f"\n{test['id']} "
                f"{test['question']}"
            )

            result = evaluate_case(
                test,
                top_k
            )

            all_results.append(
                result
            )

            print(
                "Correct:",
                result[
                    "behavior_correct"
                ]
            )

            print(
                "Latency:",
                result[
                    "latency_seconds"
                ]
            )

    output_file = (
        BASE_DIR / "rag_evaluation_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "should_answer",
                "top_k",
                "abstained",
                "behavior_correct",
                "latency_seconds",
                "answer"
            ]
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print(
        "\nResults saved to:",
        output_file
    )

    print(
        "\nSUMMARY"
    )

    print("=" * 60)

    for top_k in [1, 3, 5]:

        subset = [
            r
            for r in all_results
            if r["top_k"] == top_k
        ]

        correct = sum(
            r["behavior_correct"]
            for r in subset
        )

        accuracy = (
            correct
            / len(subset)
            * 100
        )

        avg_latency = sum(
            r["latency_seconds"]
            for r in subset
        ) / len(subset)

        print(
            f"\nTop K = {top_k}"
        )

        print(
            f"Behavior Accuracy: "
            f"{accuracy:.1f}%"
        )

        print(
            f"Average Latency: "
            f"{avg_latency:.2f}s"
        )


if __name__ == "__main__":
    main()