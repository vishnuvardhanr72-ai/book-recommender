import json

import pytest

from service.book_service import BookNotFoundError, BookService


class FakeEmbeddingModel:
    """
    Fake 3-number vectors instead of real 1536-number ones:
    [magic?, space?, habits?]  -> 1.0 if the word appears, else 0.0
    No API calls, no cost, always the same answer.
    """

    model = "fake"
    keywords = ["magic", "space", "habit"]

    def embed(self, text):
        return self.embed_all([text])[0]

    def embed_all(self, texts):
        return [[float(k in t.lower()) for k in self.keywords] for t in texts]


@pytest.fixture
def service(tmp_path):
    """Build a BookService with 4 tiny fake books before each test."""
    books = [
        {"title": "Wizard School", "description": "A boy learns magic at school."},
        {"title": "Dragon Magic", "description": "A quest full of magic and dragons."},
        {"title": "Mars Survival", "description": "An astronaut alone in space."},
        {"title": "Tiny Habits", "description": "How to build a good habit."},
    ]
    books_file = tmp_path / "books.json"
    books_file.write_text(json.dumps(books))
    svc = BookService(FakeEmbeddingModel(), books_file)
    svc.initialize_books()
    return svc


# ---------- COMPARE ----------
def test_cosine_identical_is_one():
    assert BookService.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_unrelated_is_zero():
    assert BookService.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_safe():
    assert BookService.cosine_similarity([0, 0], [1, 1]) == 0.0


# ---------- Feature 1: search ----------
def test_search_finds_magic_books(service):
    results = service.search("I want something with magic", top_k=2)
    assert {r.title for r in results} == {"Wizard School", "Dragon Magic"}


def test_search_respects_top_k(service):
    assert len(service.search("magic", top_k=1)) == 1


# ---------- Feature 2: similar books ----------
def test_similar_books_excludes_itself(service):
    results = service.similar_books("wizard school")  # lower-case on purpose
    assert results[0].title == "Dragon Magic"
    assert all(r.title != "Wizard School" for r in results)


def test_unknown_book_raises(service):
    with pytest.raises(BookNotFoundError):
        service.similar_books("Unknown Book")


# ---------- STORE ----------
def test_embeddings_are_cached(tmp_path):
    books_file = tmp_path / "books.json"
    books_file.write_text(json.dumps([{"title": "A", "description": "magic"}]))
    cache_file = tmp_path / "cache.json"

    first = BookService(FakeEmbeddingModel(), books_file, cache_file)
    first.initialize_books()
    assert cache_file.exists()

    class FailingModel(FakeEmbeddingModel):
        def embed_all(self, texts):
            raise AssertionError("should have used the cache!")

    second = BookService(FailingModel(), books_file, cache_file)
    second.initialize_books()  # must NOT call the model
    assert second.books[0].embedding == first.books[0].embedding