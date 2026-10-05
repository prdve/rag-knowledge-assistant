# GenAI Knowledge Brain

A local Retrieval-Augmented Generation (RAG)
system that answers questions using a custom
AI knowledge base.

## Features

- Local document ingestion
- Text chunking with overlap
- Sentence Transformer embeddings
- Chroma vector database
- Semantic similarity search
- Retrieval-Augmented Generation
- Ollama local LLM
- Grounded answers
- Source visibility
- Hallucination / abstention testing
- Top-K retrieval experimentation
- RAG evaluation
- Streamlit interface

## Tech Stack

- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- Ollama
- Llama 3.2 3B
- Streamlit

## Architecture

User Question
→ Query Embedding
→ Chroma Vector Search
→ Relevant Chunks
→ Context Construction
→ Ollama LLM
→ Grounded Answer + Sources

## Retrieval Experiment

The system compares retrieval using:

- Top K = 1
- Top K = 3
- Top K = 5

The experiment evaluates whether increasing
retrieved context improves answer quality or
introduces irrelevant information.

## Hallucination Mitigation

The LLM is instructed to answer only using
retrieved context.

When sufficient information is unavailable,
the assistant abstains instead of inventing
an answer.

## Run

Install dependencies:

```bash
pip install -r requirements.txt