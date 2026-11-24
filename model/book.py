from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    """A book plus its embedding vector (the list of numbers that represents its meaning)."""

    title: str
    description: str
    embedding: list[float]


@dataclass(frozen=True)
class BookMatch:
    """One search result: a book and how similar it is to the query (0 to 1)."""

    title: str
    description: str
    score: float