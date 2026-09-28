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

## Rung 2 — control forms (`forms.py`)

**Forms are read**: 39 of them from the compiler's callback declarations
(`map`, `filter`, `some`, `every`, `find`, `findIndex`, `sort`, `reduce`,
`reduceRight`, with and without an initial value), each callback a hole whose
scope and type are its declared parameters and result; the ternary added as
the language's own form. **Holes are subgoals**: `map` and `filter` push the
spec down one element at a time (and `map` then `join("")` for a string), and
a child solver sharing the parent's memory fills the hole; the other forms are
grown with small bodies, compete for budget in the same attention queue, and
are checked whole — which also puts forms inside compositions
(`xs.filter(...).length`).

What the forms forced, all general:

- **Functional dependence as promise.** `x % 2` resembles neither `true`
  nor `false`, so attention ignored it; now a value the output is a function
  of is promising, whatever its type (`search.determines`).
- **A composition is as promising as its best part**, not the mean: `=== 0`
  beside it costs nothing.
- **A constant recurs; data varies.** Taking every example value as a
  literal flooded the pool (a sub-spec's elements) and let outputs be
  memorised (`s.replace("aB", "Ab")`); now only values in at least half the
  examples, numbers and single characters, at most three.
- **Subgoals after the cheap level; Occam among receivers**: deduction
  waits until the first level is grown, and offers `xs` before
  `xs.filter(...)`.
- **The checker survives what it runs**: a heap cap, and a batch that kills
  Node is halved until the one that does is alone.

Examples it finds by deduction: `xs.filter((x, i) => ((x % 2) === 0))`,
`s.split("").map((x, i) => x.toUpperCase().replace(x, x.toLowerCase()))
.join("")` (swap case), `words.map((x, i) => x.length)`.

**Measured** (budget 8,000; train 30 then test 30; the generated tasks now
include the ternary, so rung-1 rows are not the table above):

| | dev solved / general | **held solved / general** | held evaluated |
|---|---|---|---|
| rung 2: baseline | 9 / 9 | 8 / 6 | 166k |
| rung 2: meet + repair | 16 / 15 | 13 / 8 | 197k |
| rung 2: meet + repair + **forms** | 21 / 19 | **21 / 19** | **120k** |
| rung 1: meet + repair | 30 / 26 | 27 / 21 | 51.6k |
| rung 1: meet + repair + forms | 29 / 26 | 28 / 21 | 70.1k |

**HumanEval-TS** (151, budget 8,000, judged by its own tests): meet +
repair passes **6**; with forms **13**, 12 of those routes by deduction —
the prompt's examples met for 68. The gap between 68 and 13 is still the
examples: two or three do not pin a function down, which the meaning encoder
(reading the prompt's English, `PLAN.md`) is for.

On rung 2's held tasks forms more than double what generalises (8 → 19) for
40% fewer candidates. On rung 1 they cost 37% more candidates and change
nothing that matters: the one regression left.

## Reading English and code together (`meaning.py`, `reader.py`)

One encoder reads one sequence — the English, then whatever code there is
(signature, examples, a body) — into one `Meaning`: the types, behaviour
predicates, what the program uses, what gives its result. The exact
channels teach and check it and are never asked at run time: the compiler
reads what code uses (`tscheck.js` `structure`), running reads behaviour
(MultiPL-E's tests are pairs). Every record is taught in several views of
its one sequence with the same target (`teach_meaning.py`):

- 200 `lib.d.ts` members with their JSDoc;
- MBPP-TS, solved by SmolLM3 offline and kept by the tests (257/375);
- 4,000 generated programs said in English by SmolLM3, kept when code
  rewritten from the English alone behaves the same (1,290 kept);
- HumanEval-TS never taught from (solved 99/158 the same way, to measure
  `uses` with).

**Bases, chosen on MBPP dev** (behaviour F1 / uses R@10, English +
signature + examples): MiniLM 0.49 / 0.39, ModernBERT 0.65 / 0.61,
**UniXcoder 0.67 / 0.61** → `llm/meaning-unixcoder`.

**Together against either alone** (UniXcoder, HumanEval held):

| view | returns | behaviour F1 | uses R@10 | root |
|---|---|---|---|---|
| English alone | 0.77 | 0.51 | 0.37 | 0.31 |
| code alone (signature + examples) | 0.91 | 0.59 | 0.44 | 0.42 |
| English + signature | 0.91 | 0.58 | **0.49** | 0.39 |
| English + signature + examples | 0.91 | **0.60** | 0.48 | 0.41 |

Together reads best, but by little over code alone: the examples carry
most of the behaviour, and the English adds most to *what the program is
made of*.

**The round trip at search time**: a program that meets the examples is
run on inputs varied from them and read back into behaviour; it is refused
if it lacks what the reading is sure of. Two things learned on dev:
facts about the output's range (a whole number, no repeats, can be empty,
case) are true of a task's test inputs, not of its function — varied
inputs break them for the right program — so only relations to the inputs
and order are checked (`meaning.checkable`); and the reader is rarely sure
of those, and often wrong when it is: only at 0.95 does it refuse no
right program on dev, and there it is sure of one. The gate is sound and
nearly idle.

**HumanEval-TS** (151, budget 8,000, meet + repair + forms, held run once):
passes its tests **13 → 16**, examples met 68 = 68, candidates 1.24M →
1.22M, refused 0. The gain is all the reading's `uses` as promise in the
attention queue: among programs that meet two or three examples, the one
grown first is more often the one the English asked for.

## Writing: the decoder as proposer (`parse.py`, `teach_sketch.py`, `sketcher.py`)

**Code is read back into the search's trees** (`parse.py`, over the
compiler's typed AST, `tscheck.js` `tree`): every node must be a library
operator or form with the type the compiler gives it, callback parameters
renamed to the form's own; anything else is not a program the search could
have built, and is dropped. 997/1,000 generated programs read back exactly.

**The decoder** (SmolLM2-360M → `llm/sketcher-e2`) is taught, from the
request and the meaning the reader read in it, to write one expression in
that language: 3,994 generated programs, 161 library members, and 77 MBPP
programs SmolLM3 wrote as one expression and that passed the tests *and*
parsed (repeated ×8 in training; the only ones people asked for). What it
writes is never trusted: parsed or dropped, then admitted to the search as
candidates — one that meets the examples is taken (and round-tripped), a
near miss is repaired, its parts are in the forward trie to compose with.
A proposal is a sketch: what it got right is kept, the rest is searched.

The decoder alone (greedy + 8 samples), chosen on MBPP dev:

| | MBPP dev (83): passes tests | HumanEval held (158): passes tests |
|---|---|---|
| 4 epochs, with meaning | 14 | 12 |
| 4 epochs, without | 13 | 11 |
| **2 epochs, with meaning** | **20** | 13 |
| 2 epochs, without | 20 | 12 |

**The meaning does not help the decoder** — a tie on dev. It already reads
the request itself; a line summarising the reader's reading adds nothing it
could not see. Reported as that.

**HumanEval-TS with the search** (151, budget 8,000, held, run once):

| | passes its tests | examples met | candidates |
|---|---|---|---|
| meet + repair + forms | 13 | 68 | 1.24M |
| + the reading's `uses` as promise | 16 | 68 | 1.22M |
| + the decoder's sketches | **21** | **78** | 1.34M |

19 answers came straight from a proposal; the rest of the gain from
proposals' parts repaired or composed. The decoder alone passes 12–13 of
these; the search with it, 21 — neither alone does what both do.

## What is next

Rung 2 (control forms: map, filter, reduce, conditionals — callbacks as
holes, their bodies subgoals) is what HumanEval needs, and where learned
control and chunks should first have structure to learn from. Output needs
no decoder at any rung: code is printed from the tree; the English around it
("recognised from …, passes these examples") is the program's own words.
