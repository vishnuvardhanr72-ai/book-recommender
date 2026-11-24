from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # load the API key BEFORE creating the OpenAI client

from fastapi import Body, FastAPI, HTTPException, Query  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from model.book import BookMatch  # noqa: E402
from service.book_service import BookNotFoundError, BookService  # noqa: E402
from service.embedding_model import OpenAIEmbeddingModel  # noqa: E402

book_service = BookService(
    embedding_model=OpenAIEmbeddingModel(),
    books_file=BASE_DIR / "resources" / "books.json",
    cache_file=BASE_DIR / "resources" / "book_embeddings.json",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    book_service.initialize_books()  # runs once when the server starts
    yield


app = FastAPI(title="Book Recommendation API", lifespan=lifespan)


@app.get("/api/books")
def list_books() -> list[dict[str, str]]:
    """All books (title + description) - used by the frontend's 'All books' row."""
    return [{"title": b.title, "description": b.description} for b in book_service.books]


@app.post("/api/books/search")
def search(
        query: str = Body(..., media_type="text/plain"),
        top_k: int = Query(3, ge=1, le=10),
) -> list[BookMatch]:
    """Feature 1: search by meaning, e.g. 'an exciting adventure with magic'."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text is empty.")
    return book_service.search(query, top_k)


@app.get("/api/books/similar")
def similar(
        title: str = Query(..., description="e.g. The Hobbit"),
        top_k: int = Query(3, ge=1, le=10),
) -> list[BookMatch]:
    """Feature 2: 'If you liked X, you'll like...'"""
    try:
        return book_service.similar_books(title, top_k)
    except BookNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Frontend: serves static/index.html at http://127.0.0.1:8004/
# Mounted LAST so the /api/... routes above are matched first.
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="frontend")


# Lets the green ▶ Run button start the server on port 8004
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8004)