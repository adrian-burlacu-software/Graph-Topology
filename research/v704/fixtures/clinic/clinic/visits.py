"""Visits booked with a doctor."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from clinic.patients import Patient, is_senior

#: what a visit costs
VISIT_PRICE = 40
#: days ahead a visit may be booked
BOOK_AHEAD = 30
#: the most visits a doctor sees in a day
DAILY_LIMIT = 8


class DoctorFull(Exception):
    """A doctor has no room left that day."""


@dataclass
class Visit:
    patient: Patient
    doctor: str
    day: date

    def price(self) -> float:
        return price_for(self.patient)

    def reminder(self) -> date:
        return self.day - timedelta(days=1)


def price_for(patient: Patient) -> float:
    if is_senior(patient):
        return VISIT_PRICE / 2
    return VISIT_PRICE


def cancel_fee(hours: int) -> int:
    """What cancelling `hours` before a visit costs."""
    if hours < 0:
        raise ValueError("a visit is not cancelled after it")
    return 0 if hours >= 24 else 15


def book(patient: Patient, doctor: str, booked: list, day: date) -> Visit:
    if len([one for one in booked
            if one.doctor == doctor and one.day == day]) >= DAILY_LIMIT:
        raise DoctorFull(doctor)
    return Visit(patient, doctor, day)


def on_day(visits: list, day: date) -> list:
    return [one for one in visits if one.day == day]
