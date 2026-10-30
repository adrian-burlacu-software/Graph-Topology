"""The tools the shop hires out."""
from __future__ import annotations

import csv
from dataclasses import dataclass

#: what a tool costs a day where its row says nothing
DAILY_FEE = 12
#: the deposit asked, as a share of the tool's value
DEPOSIT_SHARE = 0.25


@dataclass
class Tool:
    code: str
    name: str
    group: str
    value: int
    fee: int


def read_tools(path: str = "data/tools.csv") -> list:
    with open(path, encoding="utf-8") as handle:
        return [Tool(row["code"], row["name"], row["group"],
                     int(row["value"]), int(row["fee"]))
                for row in csv.DictReader(handle)]


def of_group(tools: list, group: str) -> list:
    return [tool for tool in tools if tool.group == group]


def dearest(tools: list):
    """The tool of the highest value."""
    return max(tools, key=lambda tool: tool.value)


def deposit(tool: Tool) -> float:
    return tool.value * DEPOSIT_SHARE


def clean_code(code: str) -> str:
    """A tool's code in capitals with its spaces gone."""
    return code.replace(" ", "").upper()
