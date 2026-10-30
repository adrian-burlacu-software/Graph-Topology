"""The shop's small web answer and command line."""
from __future__ import annotations

import sys

from rental.tools import dearest, read_tools

BUILD = "3.4.1"
PAGES = {"/tools": "tools", "/hours": "hours"}


def page(path: str) -> str:
    if path == "/tools":
        return ", ".join(tool.name for tool in read_tools())
    if path == "/hours":
        return "9 to 5"
    return "not found"


def main(argv: list) -> int:
    if argv[:1] == ["dearest"]:
        print(dearest(read_tools()).name)
        return 0
    if argv[:1] == ["build"]:
        print(BUILD)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
