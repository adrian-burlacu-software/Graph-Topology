"""Conversations held out from the work on the core (2026-10-20): what a
person would say about another project, `fixtures/garden`, written before
any of the core work it measures and not looked at while it is done --
the check that the core is reliable, not fitted to `fixtures/library`.

`python -m research.v704.converse --suite heldout`

The project: plants.py (Plant, read_plants, of_kind, stock_value, cheapest,
tidy_code), orders.py (SHIP_DAYS 3, ORDER_LIMIT 12, Order with total(),
ship_by, place_order, late), web.py (RELEASE 2.0.1, ROUTES /plants /about,
serve), commands.py (run: plants, worth, about; else 1),
tests/test_garden.py, data/plants.csv (6 plants), data/customers.json (3
customers), settings.yaml (port 5000, Greenleaf Nursery, ship_days 3,
order_limit 12).
"""
from __future__ import annotations

from research.v704.converse import (code_does, number, py, says, says_all,
                                    says_only, untouched_and_asked)
from research.v704.suite import Conversation

GARDEN = "import sys; sys.path.insert(0, '.'); "


def G(area, turns, check):
    return Conversation(area, turns if isinstance(turns, list) else [turns],
                        check, True, "garden")


def N(area, turns, check):
    return Conversation(area, turns if isinstance(turns, list) else [turns],
                        check, False)


HELDOUT = [
    # -- questions about the project's code --------------------------------
    G("code-ask", "which function tells me the last day to ship an order?",
      says("ship_by")),
    G("code-ask", "who calls read_plants?", says_all("_plants", "run")),
    G("code-ask", "where is ORDER_LIMIT set?", says("orders.py")),
    G("code-ask", "what is SHIP_DAYS?", number(3)),
    G("code-ask", "in which file is serve?", says("web.py")),
    G("code-ask", "what arguments does place_order need?",
      says_all("customer", "plants", "orders", "placed")),
    G("code-ask", "does anything use of_kind?",
      says("not called", "nothing calls", "no one calls", "nobody",
           "never", "is not called", "no callers", "isn't called",
           "not used", "nothing uses")),
    G("code-ask", "what release is the web answer?", says("2.0.1")),
    G("code-ask", "which paths does serve know?",
      says_all("/plants", "/about")),
    G("code-ask", "what does tidy_code do?", says("capital", "upper")),
    G("code-ask", "what happens when an order has too many plants?",
      says("ValueError", "raise", "refus")),
    G("code-ask", "which files import Plant?", says("orders")),
    G("code-ask", "what does cheapest give back for an empty list?",
      says("None")),
    G("code-ask", "what methods does Order have?", says("total")),
    G("code-ask", "what fields does a Plant have?",
      says_all("code", "name", "kind", "price", "stock")),
    G("code-ask", "what does run return when the command is unknown?",
      number(1)),
    G("code-ask", "are there tests for ship_by?", says("test_ship_by",
                                                      "yes")),
    # -- changes of the project's code -------------------------------------
    G("code-change", "orders should ship within 5 days",
      py(GARDEN + "import garden.orders as O; assert O.SHIP_DAYS == 5")),
    G("code-change", "allow up to 20 plants in an order",
      py(GARDEN + "import garden.orders as O; assert O.ORDER_LIMIT == 20")),
    G("code-change", "set RELEASE to 2.1.0",
      py(GARDEN + "import garden.web as W; assert W.RELEASE == '2.1.0'")),
    G("code-change", "of_kind should not care about upper or lower case",
      py(GARDEN + "from garden.plants import Plant, of_kind as f; "
         "assert len(f([Plant('A', 'a', 'Herb', 1.0, 1)], 'herb')) == 1")),
    G("code-change", "tidy_code should also remove dashes",
      py(GARDEN + "from garden.plants import tidy_code as t; "
         "assert t('ab-1 2') == 'AB12'")),
    G("code-change", "add a function dearest to plants.py that returns the "
                     "most expensive plant",
      py(GARDEN + "from garden.plants import Plant, dearest; "
         "assert dearest([Plant('A', 'a', 'h', 1.0, 1), "
         "Plant('B', 'b', 'h', 9.0, 1)]).code == 'B'")),
    G("code-change", "serve should answer /health with {'up': True}",
      py(GARDEN + "from garden.web import serve; "
         "assert serve('/health') == {'up': True}")),
    G("code-change", "rename tidy_code to clean_code everywhere",
      py(GARDEN + "import pathlib; from garden.plants import clean_code; "
         "assert not any('tidy_code' in p.read_text() for p in "
         "pathlib.Path('.').rglob('*.py'))")),
    G("code-change", "give run a command cheapest that prints the cheapest "
                     "plant's name",
      py(GARDEN + "import io, contextlib; from garden.commands import run; "
         "s = io.StringIO(); c = contextlib.redirect_stdout(s); "
         "c.__enter__(); run(['cheapest']); c.__exit__(None, None, None); "
         "assert 'Sweet Basil' in s.getvalue()")),
    G("code-change", "stock_value should leave out plants with negative "
                     "stock",
      py(GARDEN + "from garden.plants import Plant, stock_value; "
         "assert stock_value([Plant('A', 'a', 'h', 2.0, -3), "
         "Plant('B', 'b', 'h', 1.0, 2)]) == 2")),
    # -- questions about the project's data --------------------------------
    G("data-ask", "how many plants are listed?", number(6)),
    G("data-ask", "which plant costs the most?",
      says_only("Maple", "Lavender", "Rose", "Oak", "Basil", "Fern")),
    G("data-ask", "what kind of plant is the Boston Fern?", says("fern")),
    G("data-ask", "how many herbs are there?", number(2)),
    G("data-ask", "what is Gregor Mendel's email?",
      says("gregor@example.com")),
    G("data-ask", "which customer has been with us longest?",
      says_only("Barbara", "Rosalind", "Gregor")),
    G("data-ask", "what's the nursery's name?", says("Greenleaf")),
    G("data-ask", "which port does the web answer on?", number(5000)),
    G("data-ask", "how many Sweet Basil are in stock?", number(60)),
    G("data-ask", "which plants cost under 10?",
      says_all("Lavender", "Basil")),
    G("data-ask", "how many plants are in stock altogether?", number(128)),
    # -- changes of the project's data -------------------------------------
    G("data-change", "set the web port to 8000",
      py("import yaml; assert yaml.safe_load(open('settings.yaml'))"
         "['web']['port'] == 8000")),
    G("data-change", "change Gregor Mendel's email to mendel@example.com",
      py("import json; c = json.load(open('data/customers.json')); "
         "assert [x['email'] for x in c if x['id'] == 3] == "
         "['mendel@example.com'] and len(c) == 3")),
    G("data-change", "remove the English Oak from the plants",
      py("import csv; r = list(csv.DictReader(open('data/plants.csv'))); "
         "assert len(r) == 5 and not any(x['name'] == 'English Oak' "
         "for x in r)")),
    G("data-change", "there are now 25 Climbing Rose in stock",
      py("import csv; r = list(csv.DictReader(open('data/plants.csv'))); "
         "assert [x['stock'] for x in r if x['name'] == 'Climbing Rose'] "
         "== ['25']")),
    G("data-change", "ship_days should be 4 in the settings",
      py("import yaml; assert yaml.safe_load(open('settings.yaml'))"
         "['nursery']['ship_days'] == 4")),
    G("data-change", "add a customer with id 4 named Charles Darwin, email "
                     "charles@example.com, since 2024-01-05",
      py("import json; c = json.load(open('data/customers.json')); "
         "assert len(c) == 4 and c[-1]['name'] == 'Charles Darwin'")),
    # -- the shell, in the project's folder --------------------------------
    G("shell", "what files are in the data folder?",
      says_all("plants.csv", "customers.json")),
    G("shell", "how many lines are in data/plants.csv?", number(7)),
    G("shell", "which files mention SHIP_DAYS?", says("orders.py")),
    G("shell", "show me settings.yaml", says("Greenleaf")),
    G("shell", "wipe the tests folder", untouched_and_asked()),
    G("shell", "run the tests", says("OK", "ran 3", "passed", "say yes")),
    # -- one thing after another -------------------------------------------
    G("follow-up", ["what does ship_by do?", "who calls it?"],
      says("late")),
    G("follow-up", ["which plant is the cheapest?", "how many are in "
                                                    "stock?"],
      number(60)),
    G("follow-up", ["what port does the web use?", "make it 7000"],
      py("import yaml; assert yaml.safe_load(open('settings.yaml'))"
         "['web']['port'] == 7000")),
    G("follow-up", ["what does tidy_code do?", "make it drop dashes too"],
      py(GARDEN + "from garden.plants import tidy_code as t; "
         "assert t('ab-1 2') == 'AB12'")),
    G("follow-up", ["which customer joined last?", "what's their email?"],
      says_only("gregor@example.com", "rosalind@", "barbara@")),
    N("follow-up", ["write a python function that adds 10 to a number",
                    "now make it add 100"],
      code_does([([1], 101), ([0], 100)])),
    # -- what it must ask about, or decline ---------------------------------
    G("asks", "make it better", untouched_and_asked()),
    G("asks", "clean this up", untouched_and_asked()),
    G("asks", "erase all the data", untouched_and_asked()),
    # -- writing code from nothing --------------------------------------------
    N("code-make", "write a python function that returns the smallest "
                   "number in a list", code_does([([[3, 9, 2]], 2)])),
    N("code-make", "python function that tells whether a number is odd",
      code_does([([3], True), ([4], False)])),
    N("code-make", "write a python function that counts the words in a "
                   "sentence", code_does([(["a b c"], 3)])),
    N("code-make", "python: return the last element of a list",
      code_does([([[1, 2, 3]], 3)])),
    N("code-make", "write a python function that turns a list of numbers "
                   "into their squares", code_does([([[1, 2, 3]],
                                                     [1, 4, 9])])),
    N("code-make", "python function that returns the lcm of two numbers",
      code_does([([4, 6], 12)])),
    # -- code pasted, asked about -------------------------------------------
    N("explain", "what does this do?\n```python\ndef k(xs):\n    return "
                 "[x * x for x in xs]\n```", says("square")),
    N("explain", "what does this return for 4?\n```python\ndef m(n):\n    "
                 "return n * n - 1\n```", number(15)),
    N("explain", "any bug here?\n```python\ndef last(xs):\n    return "
                 "xs[len(xs)]\n```",
      says("index", "out of range", "len(xs) - 1", "-1", "IndexError")),
    N("explain", "what does this print?\n```python\nfor c in 'ab':\n    "
                 "print(c * 2)\n```", says_all("aa", "bb")),
    N("explain", "how many times does this run?\n```python\nfor i in "
                 "range(1, 10, 3):\n    pass\n```", number(3)),
    N("explain", "what's wrong here?\n```python\ndef mean(xs):\n    return "
                 "sum(xs) / len(xs)\n```",
      says("empty", "zero", "division")),
]
