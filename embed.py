from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(f"Chunks loaded: {len(texts)}")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print("\nEmbeddings created successfully.")

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        f"Saved to: {EMBEDDINGS_FILE}"
    )


if __name__ == "__main__":
    main()