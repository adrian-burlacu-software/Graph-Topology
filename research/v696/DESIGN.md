# v696: cognitive search — what was built and what it measured

The contract is `PLAN.md`. This is what came of Phases 0–3, measured as it
promised: a baseline before each mechanism, dev looked at, held run once.

## Phase 0 — measure first

- **`python -m research.bench`**: one scoreboard, history in
  `research/BENCH.jsonl` by commit. Suites: planning, designer, proposer,
  actions, piqa (`--full`), and v696's code-generated, code-humaneval
  (`--full`), designer-recog, planning-recog.
- **Baseline recorded (2d914d4)**: planning blocks 20/25, delivery 20/20 (6
  shortest), errands 20/20 (13 shortest); designer 39/44, 29/34, 22/31;
  proposer **fitted 148 vs 148 — it saves nothing**: the designer imagines
  every cheaper way before Occam chooses, whatever the order, so a better
  first guess changes no work; actions 2/41.
- **The checker** (`tscheck.js`, `checker.py`): TypeScript's compiler API in
  one persistent Node process — a type check ~20 ms, candidates evaluated in
  batches. Node is the only authority on what code does.
- **Benchmarks**: MultiPL-E HumanEval-TS (158 parsed, 151 with readable
  examples) and MBPP-TS (375; its examples are its first test), fetched to
  `data/multipl-e/` (`python -m research.v696.tasks fetch`); generated tasks
  from seeds with the composition as a known answer, 5 examples shown and 5
  hidden, dev and held seeds apart.

## Phase 1 — programming as a world

- A program is a **typed tree with holes** (`program.Expr`), printed only to
  be checked. No decoder: the tree prints exactly what was verified.
- The library is **read from the compiler**, not written: every member of
  string, number, boolean and their arrays, plus globals and `Math`, with
  each optional-parameter prefix as its own operator (189), deprecated
  members dropped because `lib.d.ts` says so, callback-taking members left
  for rung 2, zero-argument calls (constants or random) not operators.
- A spec's **features** are read off its example values (types, same length,
  part of, an element of, the length of ...): what recognition walks.

## Phase 2 — the mechanisms (`cognition.py`, `search.py`)

`cognition.py` holds them with no domain in them — `Recognizer` (trie of
solved things by features, most common first), `Equivalence` (forward trie:
a repeated signature is access, not allocation), `Attention` (one queue
across every operator), `Control` (learned operator choice). The code search
uses them; so do the planner and designer (Phase 3).

| switch | what it turned out to need |
|---|---|
| meet (2b) | Type-level backward regression was too weak a second trie: the forward trie grew everything and lost to the baseline at depth 3 (5/10 vs 6/10). The backward trie had to be over **values** — how far a forward value already resembles the required output — and attention had to rank **across operators**, not within one. Then 7/10 with 30% fewer evaluations. |
| coarse (2c) | Types before values, as the reachability filter; weak on its own (nearly every type reaches every other in three steps). |
| repair (2d) | The one that pays: a near miss is walked back and one subtree swapped for a kept expression of its type. |
| learned (2e), chunks (2f), recognition (2a) | Nothing on random compositions: they share no structure to learn. That is the benchmark's limit, reported as such — no task distribution was built to favour them. |

**Generated code tasks** (depths 1–3, 30 training then 30 test, budget 5,000
candidates):

| | dev solved | dev general | **held solved** | **held general** |
|---|---|---|---|---|
| baseline (means-ends) | 23 | 18 | 19 | 13 |
| meet + repair | 27 | 19 | **27** | **20** |
| all | 26 | 19 | 25 | 18 |

Held, run once: **+8 solved, +7 that generalise**, for fewer candidates
(62.6k vs 66.4k). "All" is below meet + repair: recognition and chunks add
candidates that cost budget and win nothing here.

**HumanEval-TS** (151): the meet finds a program meeting the prompt's
examples for 41 (baseline 26), but **5 pass the real tests either way**. Two
or three examples do not pin a function down, and most of HumanEval needs
conditions and loops — rung 2.

## Phase 3 — turned back on the planner and the designer

**Designer** (v694, all three banks in sequence, recognition learning as it
goes; `design(recognizer=...)`): the way designs with the same features
ended in is tried first, and if it checks, Occam's search for something
simpler is skipped — only when everything recognition reaches agrees (the
unique-answer shortcut; a majority vote cost 3 sensible designs).

| | ways fitted | sensible | same designs |
|---|---|---|---|
| plain | 375 | 90/109 | — |
| recognising | **231 (−38%)** | 89/109 | 103/109 |

This is where the proposer's order finally saves work: a reason to trust the
first answer, not a better first guess.

**Planner** (v691, 60 sampled problems per domain, seed 696): a solved plan
kept under its problem's shape (objects renamed by role), re-bound and
checked exactly before use.

| domain | recognised | operators fired | subgoals | solved |
|---|---|---|---|---|
| delivery | 17/60 | 1221 → 1000 (−18%) | 1568 → 1270 | 60 = 60 |
| errands | 21/60 | 845 → 672 (−20%) | 3718 → 2900 | 60 = 60 |
| blocks | 4/48 | 420 → 404 | 2674 → 2662 | 34 = 34 |

## What is next

Rung 2 (control forms: map, filter, reduce, conditionals — callbacks as
holes, their bodies subgoals) is what HumanEval needs, and where learned
control and chunks should first have structure to learn from. Output needs
no decoder at any rung: code is printed from the tree; the English around it
("recognised from …, passes these examples") is the program's own words.
