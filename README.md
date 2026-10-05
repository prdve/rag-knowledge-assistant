# Local RAG Knowledge Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Ollama](https://img.shields.io/badge/LLM-llama3.2%3A3b-orange.svg)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A fast, fully local Retrieval-Augmented Generation (RAG) assistant built from scratch with local embeddings, ChromaDB vector storage, grounded prompts that strictly refuse to guess, source citations, and automated evaluation scripts.

---

## 📌 Problem Statement

Standard LLMs often make things up or guess when asked about private documents or information they were not trained on.

This project builds a simple, reliable, and completely local RAG pipeline that:
1. **Answers only from retrieved documents** rather than general training data.
2. **Prevents hallucinations** with clear system prompt boundaries and a mandatory refusal rule ("I don't have enough information in the knowledge base to answer that").
3. **Cites exact sources** by listing the source file and chunk number for every answer.
4. **Keeps data private and free** by running 100% locally with zero external API calls.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    subgraph Offline Ingestion & Indexing
        D[Raw Documents: .md, .txt, .ipynb] -->|chunk.py| C[1,250 Chunks<br/>800 chars, 150 overlap]
        C -->|embed.py| E[Dense Embeddings<br/>all-MiniLM-L6-v2, 384-dim]
        E -->|build_vectorstore.py| V[(ChromaDB Persistent Store<br/>knowledge_base collection)]
    end

    subgraph Online Query & Retrieval
        Q[User Query] -->|retrieve.py| QE[Embed Query Vector]
        QE -->|Cosine Similarity Search| V
        V -->|Top-K Chunks + Metadata| CTX[Context Assembly<br/>Formatted Sources]
    end

    subgraph Generation & Attribution
        CTX -->|rag_pipeline.py| P[Grounded System Prompt<br/>Strict Abstention Rules]
        P -->|Ollama: llama3.2:3b| LLM[Local LLM]
        LLM --> ANS[Grounded Answer + Source Citations]
        ANS --> UI[Streamlit UI / CLI Output]
    end
```

---

## 🧩 Pipeline Components

| Stage | Script | Technology | Description |
|---|---|---|---|
| **1. Chunking** | [chunk.py](chunk.py) | Python, JSON | Parses Markdown, text, and Jupyter Notebook cells, splitting them into 800-character chunks with a 150-character overlap to keep context intact across boundaries. Produces 1,250 indexed chunks. |
| **2. Embeddings** | [embed.py](embed.py) | `sentence-transformers` | Generates 384-dimensional dense vectors using `all-MiniLM-L6-v2` and saves them to disk (`embeddings.npy`). |
| **3. Vector Store** | [build_vectorstore.py](build_vectorstore.py) | `chromadb` | Creates and populates a local ChromaDB database (`vector_store/`) indexing chunk text, IDs, and metadata (file path and chunk number). |
| **4. Retrieval Engine** | [retrieve.py](retrieve.py) | `chromadb`, PyTorch | Encodes the user's query, runs a nearest-neighbor cosine search against ChromaDB, and returns the top matching chunks with distance scores. |
| **5. Grounded RAG** | [rag_pipeline.py](rag_pipeline.py) | `ollama` (`llama3.2:3b`) | Builds the context block, injects anti-hallucination rules, runs the local LLM at `temperature=0`, and returns the attributed answer. |
| **6. Web Interface** | [app.py](app.py) | `streamlit` | Clean, interactive web UI with real-time Top-K tuning (1–5), response latency tracking, and collapsible source evidence inspectors. |

---

## 🛡️ Anti-Hallucination & Grounding Rules

The system prompt follows five practical rules to prevent false answers:
1. **Use Only Provided Context:** The model must not use outside assumptions to answer questions about the knowledge base.
2. **Say "I Don't Know":** If the retrieved chunks do not contain the answer, the model must return verbatim:
   > *"I don't have enough information in the knowledge base to answer that."*
3. **No Guessing or Extrapolation:** If a detail is missing, state that it is missing rather than filling in blanks.
4. **Source Citations:** Every valid answer must list the source numbers used (e.g. `Sources used: [1, 2]`).
5. **Deterministic Settings (`temperature=0`):** Runs the model with zero temperature so answers remain consistent and factual.

---

## 🔬 Evaluation & Benchmarks

The project includes two evaluation scripts to test answer accuracy, abstention behavior, and Top-K tradeoffs:

### 1. Behavior & Abstention Test (`eval_rag.py`)
Tests 10 curated questions across different Top-K values (1, 3, and 5):
- **6 In-Scope Questions** (e.g. tokenization, vector databases, chunking strategies) to confirm the model answers accurately when context is present.
- **4 Out-of-Scope Questions** (e.g. sports, cooking, general trivia) to verify the model correctly says it does not have enough information.

### 2. Retrieval Attribution & Top-K Parameter Study (`rag_test_cases.py`)
Measures how retrieval performance shifts across Top-K (1, 3, 5):
- **Abstention Accuracy:** How reliably out-of-scope questions trigger a refusal.
- **Source Match Rate:** How often the top-retrieved chunks match the intended document.
- **Noise vs. Context:** How extra chunks impact prompt size and answer focus as $K$ increases.
- **Latency:** End-to-end response time for embedding, retrieval, and generation.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- Pull the 3B model:
  ```bash
  ollama pull llama3.2:3b
  ```

### 2. Installation
```bash
git clone https://github.com/prdve/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

### 3. Pipeline Indexing (Pre-built artifacts included)
*Note: The repository already includes indexed `chunks.json`, `embeddings.npy`, and `vector_store/`. To rebuild everything from scratch:*
```bash
# 1. Extract and chunk documents
python chunk.py

# 2. Compute vector embeddings
python embed.py

# 3. Build ChromaDB collection
python build_vectorstore.py
```

### 4. Running the Assistant
#### Web Interface (Streamlit)
```bash
streamlit run app.py
```

#### Terminal CLI Mode
```bash
python rag_pipeline.py
```

#### Run Automated Evaluation
```bash
python eval_rag.py
python rag_test_cases.py
```

---

## 📁 Repository Structure

```
rag-knowledge-assistant/
├── chunk.py                  # Document ingestion and character-overlap chunker
├── embed.py                  # Batch vector embedding with all-MiniLM-L6-v2
├── build_vectorstore.py      # ChromaDB persistent collection loader
├── retrieve.py               # Semantic query encoder and cosine search engine
├── rag_pipeline.py           # Grounded prompt assembly and Ollama inference
├── app.py                    # Streamlit web application with source inspectors
├── eval_rag.py               # Behavioral evaluation suite (answer vs. abstain)
├── rag_test_cases.py         # Advanced evaluation (source match & Top-K tuning)
├── requirements.txt          # Python dependencies
├── chunks.json               # Serialized chunk corpus (1,250 chunks)
├── embeddings.npy            # Precomputed embedding matrix (1250 x 384)
├── vector_store/             # Persistent ChromaDB SQLite & segment files
└── Data/                     # Raw knowledge documents (.md, .txt, .ipynb)
    └── Content/              # Educational modules (tokens, embeddings, RAG, LLMs)
```

---

## 📄 License

This project is licensed under the MIT License.
