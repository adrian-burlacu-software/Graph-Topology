"""What the library answers over HTTP: its books and how it is."""
from __future__ import annotations

from library import books as B

VERSION = "1.2.0"


def handle(path: str) -> dict:
    """The answer to a GET of `path`."""
    if path == "/api/books":
        return {"books": [book.title for book in B.load_books()]}
    if path == "/api/status":
        return {"name": "library", "version": VERSION}
    return {"error": f"no such page: {path}"}
