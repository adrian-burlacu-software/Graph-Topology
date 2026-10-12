"""Who has borrowed what, and when it is due back."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from library.books import Book

#: how long a book may be kept
LOAN_DAYS = 14
#: how many books a member may hold at once
MAX_LOANS = 3


@dataclass
class Loan:
    book: Book
    member: str
    start: date


def due_date(start: date) -> date:
    """When a loan begun on `start` must be returned."""
    return start + timedelta(days=LOAN_DAYS)


def lend(book: Book, member: str, loans: list, start: date) -> Loan:
    """A new loan of `book` to `member`, refused past MAX_LOANS."""
    held = [loan for loan in loans if loan.member == member]
    if len(held) >= MAX_LOANS:
        raise ValueError(f"{member} already holds {MAX_LOANS} books")
    loan = Loan(book, member, start)
    loans.append(loan)
    return loan


def overdue(loans: list, today: date) -> list:
    """The loans past their due date."""
    return [loan for loan in loans if due_date(loan.start) < today]
