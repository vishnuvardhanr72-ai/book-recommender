import json
import math
from pathlib import Path

from model.book import Book, BookMatch


class BookNotFoundError(LookupError):
    pass


class BookService:
    """
    REPRESENT -> STORE -> COMPARE -> RANK
    """

    def __init__(self, embedding_model, books_file: Path, cache_file: Path | None = None):
        self.embedding_model = embedding_model
        self.books_file = Path(books_file)
        self.cache_file = Path(cache_file) if cache_file else None
        self.books: list[Book] = []

    # ---------- REPRESENT + STORE ----------
    def initialize_books(self) -> None:
        with self.books_file.open(encoding="utf-8") as f:
            raw_books = json.load(f)

        texts = [f"{b['title']}. {b['description']}" for b in raw_books]

        embeddings = self._load_cache(texts)
        source = "loaded from cache"
        if embeddings is None:
            embeddings = self.embedding_model.embed_all(texts)  # ONE API call for all books
            self._save_cache(texts, embeddings)
            source = "computed"

        self.books = [
            Book(b["title"], b["description"], e) for b, e in zip(raw_books, embeddings)
        ]
        print(f"{len(self.books)} books {source} (vector size {len(self.books[0].embedding)}).")

    def _load_cache(self, texts: list[str]) -> list[list[float]] | None:
        if not self.cache_file or not self.cache_file.exists():
            return None
        cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
        if cached.get("model") == getattr(self.embedding_model, "model", None) and cached.get("texts") == texts:
            return cached["embeddings"]
        return None  # books changed -> recompute

    def _save_cache(self, texts: list[str], embeddings: list[list[float]]) -> None:
        if not self.cache_file:
            return
        self.cache_file.write_text(
            json.dumps({
                "model": getattr(self.embedding_model, "model", None),
                "texts": texts,
                "embeddings": embeddings,
            }),
            encoding="utf-8",
        )

    # ---------- COMPARE ----------
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """cos(theta) = (a . b) / (|a| * |b|)   ->  1 = same meaning, 0 = unrelated"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    # ---------- RANK ----------
    def _rank(self, target: list[float], top_k: int, exclude_title: str | None = None) -> list[BookMatch]:
        matches = [
            BookMatch(b.title, b.description, round(self.cosine_similarity(target, b.embedding), 4))
            for b in self.books
            if b.title != exclude_title
        ]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    # ---------- Feature 1: semantic search ----------
    def search(self, query: str, top_k: int = 3) -> list[BookMatch]:
        query_vector = self.embedding_model.embed(query)
        return self._rank(query_vector, top_k)

    # ---------- Feature 2: similar books ----------
    def similar_books(self, title: str, top_k: int = 3) -> list[BookMatch]:
        selected = self.find_book(title)
        return self._rank(selected.embedding, top_k, exclude_title=selected.title)

    def find_book(self, title: str) -> Book:
        for book in self.books:
            if book.title.lower() == title.lower():
                return book
        raise BookNotFoundError(f"Book not found: {title}")