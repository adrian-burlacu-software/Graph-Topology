"""The library from the command line: `python -m library.cli books`."""
from __future__ import annotations

import sys

from library import books as B
from library.server import handle


def main(argv: list) -> int:
    if not argv:
        print("usage: books | pages | status")
        return 2
    if argv[0] == "books":
        for book in B.load_books():
            print(f"{book.title} ({book.author})")
    elif argv[0] == "pages":
        print(B.total_pages(B.load_books()))
    elif argv[0] == "status":
        print(handle("/api/status"))
    else:
        print(f"unknown command: {argv[0]}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
