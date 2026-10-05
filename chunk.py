import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_FILE = BASE_DIR / "chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def read_notebook(file_path):
    """Extract text from a Jupyter notebook.

    Reads both markdown and code cells and
    joins their source lines into a single
    string.
    """

    try:
        raw = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        notebook = json.loads(raw)

    except (json.JSONDecodeError, Exception) as e:
        print(f"Could not parse notebook {file_path}: {e}")
        return ""

    parts = []

    for cell in notebook.get("cells", []):

        cell_type = cell.get("cell_type", "")

        if cell_type not in ("markdown", "code"):
            continue

        source = cell.get("source", [])

        text = "".join(source).strip()

        if text:
            parts.append(text)

    return "\n\n".join(parts)


def read_documents():
    documents = []

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in (".txt", ".md"):

            try:
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

        elif suffix == ".ipynb":

            text = read_notebook(file_path)

        else:
            continue

        if text.strip():
            documents.append({
                "source": str(file_path),
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():
    documents = read_documents()

    print(f"\nDocuments found: {len(documents)}")

    all_chunks = []

    chunk_id = 0

    for document in documents:

        chunks = chunk_text(document["text"])

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "source": document["source"],
                "chunk_number": chunk_number,
                "text": chunk
            })

            chunk_id += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()