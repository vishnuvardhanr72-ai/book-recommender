# Semantic Book Recommender

A FastAPI service that recommends books by **meaning**, not keywords, using OpenAI vector embeddings and cosine similarity.

**Flow:** REPRESENT → STORE → COMPARE → RANK

## Features
- **Semantic search:** `POST /api/books/search` with plain text like *"an exciting adventure with magic"*
- **Similar books:** `GET /api/books/similar?title=The Hobbit`
- **List books:** `GET /api/books`
- Embeddings are batched into one API call and cached to disk
- 8 pytest unit tests using a fake embedding model (no API cost)

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
python book_api.py        # http://127.0.0.1:8004/docs
pytest -v
```

## Tech
Python · FastAPI · OpenAI Embeddings (`text-embedding-3-small`) · pytest