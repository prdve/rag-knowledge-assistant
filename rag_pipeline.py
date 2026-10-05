import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ollama
from retrieve import retrieve

MODEL_NAME = "llama3.2:3b"


def build_context(
    retrieved_chunks
):

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {number}
File: {item["source"]}
Chunk: {item["chunk_number"]}

{item["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


def rag_answer(
    question,
    top_k=3
):

    retrieved_chunks = retrieve(
        question,
        top_k=top_k
    )

    context = build_context(
        retrieved_chunks
    )

    system_prompt = """
You are a knowledge assistant.

You must answer using ONLY the information
contained in the provided context.

Rules:

1. Do not use outside knowledge.

2. If the answer cannot be found in the
context, say exactly:

"I don't have enough information in the
knowledge base to answer that."

3. Do not invent facts.

4. Give a clear and concise answer.

5. At the end of the answer, mention which
source numbers were used.

Example:

Sources used: [1, 3]
"""

    user_prompt = f"""
CONTEXT:

{context}


QUESTION:

{question}


Answer using only the context above.
"""

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        options={
            "temperature": 0
        }
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return {
        "question":
        question,

        "answer":
        answer,

        "sources":
        retrieved_chunks
    }


if __name__ == "__main__":

    question = input(
        "\nAsk Knowledge Brain: "
    )

    result = rag_answer(
        question,
        top_k=3
    )

    print(
        "\nANSWER\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nRETRIEVED SOURCES"
    )

    print("=" * 60)

    for number, source in enumerate(
        result["sources"],
        start=1
    ):

        print(
            f"\n[{number}] "
            f"{source['source']}"
        )

        print(
            "Chunk:",
            source["chunk_number"]
        )