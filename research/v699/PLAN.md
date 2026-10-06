# v699: Python, fully — a second language through every layer

Adrian (2026-10-05): "I want you to FULLY introduce the python language.
I'm talking re-training for python terms, the whole thing."

Branch `v699/python`, from master after v698 (`a0d27e1`).

## Where we start

Nothing in the code world knows there is more than one language. Every
check, run, parse, label and prompt goes through the TypeScript compiler
(`research/v696/tscheck.js`, one Node process) and models taught on
TypeScript only. What is already language-neutral: the search itself
(`search.py`: meet, repair, forms' deduction, agreement, ranking),
`cognition.py`, `planning.py`, the risk matrix, the behaviour predicates
(value-based), fault localisation over spans, the conversation's acts.

## The shape

**A `Language` seam** (`research/v696/language.py`): everything that is a
language's — its checker, how a tree is printed, how a function is
signed, its operators, globals and library, how a test file and a
project are made, how values are compared — behind one interface, with
`typescript` (what is there today, unchanged) and `python`. `Spec`, a
request, a project, a workspace, an answer carry `language`.

**The engine's types stay abstract** (`number`, `string`, `boolean`,
`T[]`, `[a, b]`, `fn:`): Python's `int`/`float`/`str`/`bool`/`list[T]`/
`tuple[...]`/`dict`/`set` map onto them at the checker's boundary; how a
type is *written* (`int` vs `float`) is kept for printing.

**`pycheck.py`: the Python checker**, speaking `tscheck.js`'s JSON protocol
so `checker()` becomes `checker(language)`:

| op | Python |
|---|---|
| run, values, tests, project | a worker process (its own interpreter, killed on timeout, output captured), values as JSON with the same encoding (sets, dicts, tuples, ±inf); numbers compared numerically (`2 == 2.0`) |
| check, diagnose (strict too) | **mypy** (pure Python, in process via its API; `--strict` as STRICTER) |
| signatures | the builtins and the stdlib a function would use (`str`, `list`, `dict`, `set`, `math`, `re`, `itertools`, `functools`, `collections`), read from typeshed through mypy |
| tree | `ast` + mypy's inferred types, read by symbolic execution into the same typed JSON nodes `parse.py` reads (`if`/`for`/`while`/assignments/comprehensions/`match`) — the largest single piece |
| structure, qualities, shape, outline, restyle | `ast` walks; Python's own words (`list comprehension`, `str.split`, `for in range`, `recursion`) |

**Printing a tree as Python** (`Op.said`, `Expr.text` by language):
`a if c else b`, `lambda x: …`, comprehensions for map/filter,
`functools.reduce`, `sep.join(xs)` (the receiver swaps), `range`, helper
`def`s for what TypeScript prints as IIFEs (`let`, `while`, effects).

## The models, re-taught for Python

**Bilingual** (Adrian, 2026-10-05): one model per role, taught TypeScript
and Python together, the language in its prompt — Python borrows what the
model knows of TypeScript, one set is loaded. Where a role's TypeScript
numbers fall beyond noise of today's, that role falls back to a model of
Python's own.

Every model in the code world is taught again on Python as well, each into
a new `llm/` directory (the trained ones are untouchable), every corpus
regenerable by `python -m regenerate`:

| model | today | for Python |
|---|---|---|
| datasets | MultiPL-E `mbpp-ts`, `humaneval-ts`; HumanEvalPack `js` | `mbpp-py`, `humaneval-py`, the originals' reference solutions, HumanEvalPack `python` — same names, so the same dev and held splits |
| code-meaning corpora (SmolLM3, offline) | docs of lib.d.ts, solutions, compositions described | builtins'/stdlib docstrings run on values, Python solutions (kept where the tests pass), Python compositions described |
| reader of meaning (`meaning-unixcoder`) | `uses`/`root` are TS operators, `returns` TS types | the same heads over a vocabulary of both languages' words; UniXcoder already reads Python |
| writers (`sketcher-*`, SmolLM2-360M) | "You write TypeScript", TS targets | taught to write Python too: the language in the system line, Python targets |
| invented requests (`teach_requests`) | TS tasks | Python tasks, solved, kept where they meet their own examples |
| risk estimators | labels from tscheck `qualities` | labels from `pycheck` qualities; U from Python samples |
| ways (`ways.py`, `ways-estimator`) | 28 JS/TS ways | Python's own ways beside them: comprehension, generator, f-string, lambda, `match`, `with`, `enumerate`/`zip`, unpacking, `functools.reduce`, `dict`/`set`, `def` vs `lambda`, recursion/loops; **which language** a message asks for (*in Python*) read by the same encoder (a `language` family) — never a word list |
| code talk (`teach_code_talk`, shared reader) | camelCase names, `.ts` files, JS techniques | snake_case names (Python's own), `.py` files, Python techniques; the reader taught again from `reader-code9`, the suite the gate |

## The conversation, the project, the editor

- **Reading a request**: `def f(x: int) -> int:`, examples `f(1) == 2` and
  `assert f(1) == 2`, Python literals (`True`, `None`, tuples) read by
  Python; a function written out (indentation, not braces) is the
  person's own, as in v698.
- **Which language**: the code written (a `def` is Python), the project's
  files, the editor's language, what the message says (the encoder) — else
  the conversation's, else TypeScript.
- **Knowledge**: a taught concept is offered to the search as `def
  is_vowel(x): return x in (...)`.
- **The project**: `.py` files, packages and relative imports, `__init__`,
  `self.`, mypy's diagnostics (strict), the outline from `ast`.
- **Asking**: paths `.py`, Python's signatures, mypy's codes, *gives
  nothing (None)* where its type promises a value.
- **The page**: Python highlighting, `PY` label, `.py` downloads, insert.
- **The extension**: `**/*.py` read too, `__pycache__`/`.venv` excluded,
  Python language ids, new documents opened as Python.

## Phases (each committed when its gate holds)

1. **The seam** — `Language`, `checker(language)`, TypeScript behind it.
   Gate: the suite, and HumanEval-TS held / MBPP dev unchanged (quick eval).
2. **pycheck** — run, values, tests, check/diagnose, signatures, structure,
   qualities, shape, outline. Gate: tests of each op; MBPP-py reference
   solutions all pass their asserts through it.
3. **The Python tree and printer** — `tree`, the library from typeshed,
   printing. Gate: MBPP-py reference solutions read into trees and printed
   back pass their tests (the TypeScript reader reads 77/375 as one
   expression; the whole-function rate is the bar to match).
4. **Datasets and the search on Python, untaught** — fetch, splits, the
   search with the untaught writer. Gate: MBPP-py dev / HumanEval-py held
   numbers recorded (the baseline to beat).
5. **Corpora and models re-taught** — code-meaning, reader of meaning,
   writers, requests, risk. Gate: Python numbers above phase 4's;
   TypeScript's within noise of today's.
6. **Ways and code talk** — Python ways, the language family, code talk
   re-taught, the shared reader re-taught. Gate: the suite; ways ≥ 80%
   precise per way in both languages (Adrian's bar).
7. **Conversation, project, page, extension** — language through every
   layer. Gate: tests; over HTTP a Python conversation end to end (request,
   change with a way, bugs, your own function, a project of `.py` files).
8. **Measure and document** — HumanEvalFix-python, every number, DESIGN.md.

Costs (GPU, offline teacher): corpora ~4 h, writers ~1.5 h, requests ~2 h,
risk labels ~1.5 h, code talk ~3 h, readers ~1 h — about two days of
runs beside the engineering.
