"""The library's books, read from data/books.csv."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Book:
    id: int
    title: str
    author: str
    pages: int
    year: int


def load_books(path: Path = DATA / "books.csv") -> list:
    """Every book of the catalogue, in the file's order."""
    with open(path, encoding="utf-8") as stream:
        return [Book(int(row["id"]), row["title"], row["author"],
                     int(row["pages"]), int(row["year"]))
                for row in csv.DictReader(stream)]


def find_by_author(books: list, author: str) -> list:
    """The books one author wrote."""
    return [book for book in books if book.author == author]


def total_pages(books: list) -> int:
    """How many pages the books have together."""
    return sum(book.pages for book in books)


def oldest(books: list):
    """The book published first, or None for no books."""
    return min(books, key=lambda book: book.year) if books else None


def parse_isbn(text: str) -> str:
    """An ISBN with its dashes and spaces taken out."""
    return text.replace("-", "").replace(" ", "")
