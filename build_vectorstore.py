from pathlib import Path
import json
import numpy as np
import chromadb

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

VECTOR_DB_PATH = str(BASE_DIR / "vector_store")
COLLECTION_NAME = "knowledge_base"


def main():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embeddings: {len(embeddings)}"
    )

    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    # Delete old collection so rerunning this file
    # does not leave stale chunks behind.
    try:
        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old collection removed."
        )

    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
            "Zero-to-AI knowledge base"
        }
    )

    ids = []
    documents = []
    metadatas = []
    vectors = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        ids.append(
            chunk["id"]
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "source":
            chunk["source"],

            "chunk_number":
            chunk["chunk_number"]
        })

        vectors.append(
            embedding.tolist()
        )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors
    )

    print(
        "\nVector database created successfully."
    )

    print(
        "Documents stored:",
        collection.count()
    )

    print(
        f"Location: {VECTOR_DB_PATH}/"
    )


if __name__ == "__main__":
    main()