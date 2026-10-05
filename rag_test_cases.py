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
        "What is RAG?",
        "expected_source": "rag"
    },

    {
        "id": "R02",
        "question":
        "What are embeddings?",
        "expected_source": "embeddings"
    },

    {
        "id": "R03",
        "question":
        "How do embeddings help RAG?",
        "expected_source": "embeddings"
    },

    {
        "id": "R04",
        "question":
        "What is prompt engineering?",
        "expected_source": "prompt"
    },

    {
        "id": "R05",
        "question":
        "What is few-shot prompting?",
        "expected_source": "prompt"
    },

    {
        "id": "R06",
        "question":
        "What is context engineering?",
        "expected_source": "context"
    },

    {
        "id": "R07",
        "question":
        "Why does RAG use external knowledge?",
        "expected_source": "rag"
    },

    {
        "id": "R08",
        "question":
        "How does semantic search work?",
        "expected_source": "embeddings"
    },

    {
        "id": "R09",
        "question":
        "Explain something about quantum computing.",
        "expected_source": None
    },

    {
        "id": "R10",
        "question":
        "Who is the current President of "
        "the United States?",
        "expected_source": None
    }

]


ABSTENTION_TEXT = (
    "i don't have enough information "
    "in the knowledge base"
)


def evaluate_test_case(
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

        sources = result[
            "sources"
        ]

        abstained = (
            ABSTENTION_TEXT
            in answer.lower()
        )

        expected = test[
            "expected_source"
        ]

        # Source attribution check
        source_match = False
        irrelevant_count = 0

        if expected is not None:

            for chunk in sources:

                source_path = chunk[
                    "source"
                ].lower()

                if expected in source_path:
                    source_match = True
                else:
                    irrelevant_count += 1

        # Abstention correctness
        if expected is None:
            abstention_correct = abstained
        else:
            abstention_correct = (
                not abstained
            )

        # Answer quality check
        has_source_citation = (
            "sources used"
            in answer.lower()
        )

        return {

            "id":
            test["id"],

            "question":
            test["question"],

            "expected_source":
            str(expected),

            "top_k":
            top_k,

            "abstained":
            abstained,

            "abstention_correct":
            abstention_correct,

            "source_match":
            source_match,

            "irrelevant_chunks":
            irrelevant_count,

            "has_source_citation":
            has_source_citation,

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

            "expected_source":
            str(
                test["expected_source"]
            ),

            "top_k":
            top_k,

            "abstained":
            False,

            "abstention_correct":
            False,

            "source_match":
            False,

            "irrelevant_chunks":
            0,

            "has_source_citation":
            False,

            "latency_seconds":
            0,

            "answer":
            f"ERROR: {e}"

        }


def run_topk_experiment(
    question,
    k_values
):

    results = []

    for k in k_values:

        start = time.perf_counter()

        result = rag_answer(
            question,
            top_k=k
        )

        latency = (
            time.perf_counter()
            - start
        )

        results.append({

            "top_k":
            k,

            "answer":
            result["answer"],

            "sources":
            result["sources"],

            "latency":
            round(latency, 2),

            "num_chunks":
            len(result["sources"])

        })

    return results


def compare_topk_results(
    results
):

    print(
        "\nTOP-K EXPERIMENT COMPARISON"
    )

    print("=" * 70)

    for r in results:

        print(
            f"\n--- Top K = {r['top_k']} "
            f"---"
        )

        print(
            f"Chunks retrieved: "
            f"{r['num_chunks']}"
        )

        print(
            f"Latency: "
            f"{r['latency']}s"
        )

        print(
            f"\nSources:"
        )

        for number, s in enumerate(
            r["sources"],
            start=1
        ):

            print(
                f"  [{number}] "
                f"{s['source']} "
                f"(distance: "
                f"{round(s['distance'], 4)})"
            )

        print(
            f"\nAnswer:\n"
            f"{r['answer'][:500]}"
        )

        print("-" * 70)

    # Comparison summary
    print(
        "\nCOMPARISON SUMMARY"
    )

    print("=" * 70)

    print(
        f"{'Top K':<8}"
        f"{'Chunks':<10}"
        f"{'Latency':<12}"
        f"{'Avg Distance':<15}"
    )

    print("-" * 45)

    for r in results:

        distances = [
            s["distance"]
            for s in r["sources"]
        ]

        avg_dist = (
            sum(distances)
            / len(distances)
            if distances
            else 0
        )

        print(
            f"{r['top_k']:<8}"
            f"{r['num_chunks']:<10}"
            f"{r['latency']:<12}"
            f"{avg_dist:<15.4f}"
        )


def main():

    # Part 1: Run all test cases
    # at each top_k value

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

            result = evaluate_test_case(
                test,
                top_k
            )

            all_results.append(
                result
            )

            print(
                "Abstention correct:",
                result[
                    "abstention_correct"
                ]
            )

            print(
                "Source match:",
                result[
                    "source_match"
                ]
            )

            print(
                "Latency:",
                result[
                    "latency_seconds"
                ]
            )

    # Save full results to CSV

    output_file = (
        "rag_test_results.csv"
    )

    fieldnames = [
        "id",
        "question",
        "expected_source",
        "top_k",
        "abstained",
        "abstention_correct",
        "source_match",
        "irrelevant_chunks",
        "has_source_citation",
        "latency_seconds",
        "answer"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print(
        "\nResults saved to:",
        output_file
    )

    # Part 1 summary

    print(
        "\nFULL TEST SUITE SUMMARY"
    )

    print("=" * 60)

    for top_k in [1, 3, 5]:

        subset = [
            r
            for r in all_results
            if r["top_k"] == top_k
        ]

        abstention_acc = sum(
            r["abstention_correct"]
            for r in subset
        ) / len(subset) * 100

        source_acc = sum(
            r["source_match"]
            for r in subset
            if r["expected_source"] != "None"
        )

        source_total = sum(
            1
            for r in subset
            if r["expected_source"] != "None"
        )

        source_pct = (
            source_acc
            / source_total
            * 100
            if source_total > 0
            else 0
        )

        avg_latency = sum(
            r["latency_seconds"]
            for r in subset
        ) / len(subset)

        avg_irrelevant = sum(
            r["irrelevant_chunks"]
            for r in subset
        ) / len(subset)

        print(
            f"\nTop K = {top_k}"
        )

        print(
            f"  Abstention Accuracy: "
            f"{abstention_acc:.1f}%"
        )

        print(
            f"  Source Match Rate: "
            f"{source_pct:.1f}% "
            f"({source_acc}/{source_total})"
        )

        print(
            f"  Avg Irrelevant Chunks: "
            f"{avg_irrelevant:.1f}"
        )

        print(
            f"  Avg Latency: "
            f"{avg_latency:.2f}s"
        )

    # Part 2: Focused Top-K experiment

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FOCUSED TOP-K EXPERIMENT"
    )

    print(
        "Query: "
        "\"How do embeddings help RAG?\""
    )

    print("=" * 70)

    experiment_results = (
        run_topk_experiment(
            "How do embeddings help RAG?",
            [1, 3, 5]
        )
    )

    compare_topk_results(
        experiment_results
    )


if __name__ == "__main__":
    main()
