"""The front desk's small web answer and command line."""
from __future__ import annotations

import sys

from clinic.patients import eldest, load_patients

SOFTWARE = "1.9.6"
SCREENS = {"/patients": "patients", "/opening": "opening"}


def screen(path: str) -> str:
    if path == "/patients":
        return ", ".join(one.name for one in load_patients())
    if path == "/opening":
        return "8 to 6"
    return "no such screen"


def main(argv: list) -> int:
    if argv[:1] == ["eldest"]:
        print(eldest(load_patients()).name)
        return 0
    if argv[:1] == ["software"]:
        print(SOFTWARE)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
