import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

VECTOR_DB_PATH = str(BASE_DIR / "vector_store")
COLLECTION_NAME = "knowledge_base"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


model = SentenceTransformer(
    EMBEDDING_MODEL
)


client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)


collection = client.get_collection(
    COLLECTION_NAME
)


def retrieve(
    question,
    top_k=3
):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for i in range(
        len(results["documents"][0])
    ):

        retrieved.append({

            "text":
            results["documents"][0][i],

            "source":
            results["metadatas"][0][i][
                "source"
            ],

            "chunk_number":
            results["metadatas"][0][i][
                "chunk_number"
            ],

            "distance":
            results["distances"][0][i]

        })

    return retrieved


if __name__ == "__main__":

    question = input(
        "\nAsk a question: "
    )

    results = retrieve(
        question,
        top_k=3
    )

    print(
        "\nTOP RETRIEVED CHUNKS"
    )

    print("=" * 70)

    for number, item in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {number}"
        )

        print(
            "Source:",
            item["source"]
        )

        print(
            "Chunk:",
            item["chunk_number"]
        )

        print(
            "Distance:",
            round(
                item["distance"],
                4
            )
        )

        print(
            "\n",
            item["text"][:1000]
        )

        print(
            "\n" + "-" * 70
        )