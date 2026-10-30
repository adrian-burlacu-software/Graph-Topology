"""Hires of a tool by a customer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from rental.tools import DAILY_FEE, Tool, deposit

#: days a hire lasts unless asked otherwise
HIRE_DAYS = 5
#: what a day past the end costs
LATE_FEE = 7
#: the most tools one customer may hold
MAX_HIRES = 4


class TooManyHires(Exception):
    """A customer holds as many tools as they may."""


@dataclass
class Hire:
    tool: Tool
    customer: str
    start: date

    def ends(self) -> date:
        return self.start + timedelta(days=HIRE_DAYS)

    def cost(self) -> float:
        return self.tool.fee * HIRE_DAYS + deposit(self.tool)


def late_charge(days: int) -> int:
    """What a hire returned `days` late costs."""
    if days < 0:
        raise ValueError("a hire is not late by less than nothing")
    return days * LATE_FEE


def hire(tool: Tool, customer: str, held: list, start: date) -> Hire:
    if len([one for one in held if one.customer == customer]) >= MAX_HIRES:
        raise TooManyHires(customer)
    return Hire(tool, customer, start)


def plain_cost(days: int) -> int:
    return days * DAILY_FEE


def past_end(hires: list, today: date) -> list:
    return [one for one in hires if one.ends() < today]
