"""The patients a clinic has on its books."""
from __future__ import annotations

import json
from dataclasses import dataclass

#: the age from which a patient counts as senior
SENIOR_AGE = 65


@dataclass
class Patient:
    number: str
    name: str
    age: int
    ward: str


def load_patients(path: str = "data/patients.json") -> list:
    with open(path, encoding="utf-8") as handle:
        return [Patient(**row) for row in json.load(handle)["patients"]]


def in_ward(patients: list, ward: str) -> list:
    return [one for one in patients if one.ward == ward]


def eldest(patients: list):
    """The patient of the greatest age."""
    return max(patients, key=lambda one: one.age)


def is_senior(patient: Patient) -> bool:
    return patient.age >= SENIOR_AGE


def tidy_number(number: str) -> str:
    """A patient's number with its dashes gone."""
    return number.replace("-", "")
