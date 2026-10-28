# legal-rag-evals

RAG pipeline over legal documents with an evaluation harness.

Stack: `anthropic` (generation / LLM-judge), `sentence-transformers` (dense embeddings),
`rank_bm25` (sparse retrieval), `numpy` (in-memory vector store).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```
