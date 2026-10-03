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

## Rung 3 — decomposition: a body as steps

Measured first (`needs.py`): of the teacher's verified programs, 74/99
HumanEval and 153/257 MBPP use local variables and 56/133 a loop; 5 and 3
use a helper function. People decompose inside a function, so rung 3 read
the steps (`PLAN.md`).

**Bodies are executed symbolically into one tree** (`tscheck.js` `tree`,
`parse.py` `Module`) — a local name read is its value at that point:

| statement | read as |
|---|---|
| `const w = ...` (a step) | a node shared where `w` is read |
| a helper function | an operator carrying its body (`Op.body`), printed before, run as a prelude |
| `if (c) return a; ... return b` | `c ? a : b` |
| an `if` that assigns | a ternary per name it changes |
| a loop carrying one value | `reduce`: the update its body, the value before it the start |
| `for (let i = a; i < b; i++)` | a fold over `Array.from({ length: b - a }, (_, i) => a + i)` |
| a loop that returns when it finds | `xs.some(c) ? R : rest`, the element in R being `xs.find(c)` |
| `out.push(v)` in a loop | the fold's update `[...acc, v]` |

Added to the library as the language's own: element access `xs[i]`, the
range, the append, `!==` — kept out of the rung 1–2 generator (`P.LATER`),
whose dev and held task sets are checked identical.

**Reading is exact**: every verified program that reads is printed back
from its tree and passes its task's own tests (`needs_exact.py`): 88/88
MBPP, 33/33 HumanEval. What reads: MBPP 39 → **88**/257, HumanEval 14 →
**33**/99. What still does not: `while` loops and loops carrying two
values (a running maximum and a list) — tuples are beyond the library's
types — and sets, maps and objects.

**The decoder writes whole functions** (`sketcher-functions`, 2 epochs):
MBPP's verified programs as written — loops, steps — are now teachable
(127 train, from 57). Alone, MBPP dev passes 22/83 (single expressions 21)
and meets the examples for 35 (28); HumanEval held 16 (13).

**Rung-3 generated tasks** (a form over a counted range, dev):

| | solved | general |
|---|---|---|
| baseline | 1 | 0 |
| meet + repair | 3 | 2 |
| meet + repair + forms | **17** | **14** |

Deduction pushes map and filter over a range down per element (14 of 17).
The rest are folds that iterate their accumulator without the element:
only induction finds those (consecutive sizes pinning the step), which
random examples rarely give — noted, not tuned for. Rungs 1–2 dev with
the rung-3 library: 30/26 (was 29/26), 21/19 (unchanged), rung 1 at +15%
candidates.

**HumanEval-TS** (151, budget 8,000, held, once): 21 → **22** pass their
tests, examples met 78 → 86, candidates 1.34M → 1.19M.

### Rung 3, finished (2026-09-29)

- **Any loop is read.** Several values carried are a tuple (`state[1]`);
  `while` and any `for` that does not simply count are the language's own
  loop, printed as an exact expression
  (`((c, u, s) => { while (c(s)) s = u(s); return s; })(test, update,
  start)`); parameters are variables a loop may change.
- **Early exits are values.** `return`, `break` and `continue` inside a
  loop set flags the loop carries (stopped, skipped, returned and what);
  every change after such a point is guarded by them, every loop's test
  includes "not yet returned" (an outer return freezes an inner loop too —
  found as an infinite loop, fixed as a rule), and the function returns
  what was returned if anything was. The plain `some`/`find` reading of a
  search loop is kept where nothing may have returned before it.
- **Empty lists take the type they become** where the compiler next knows
  it (`var result = []`); rest parameters one by one (`Math.max(a, b)`) are
  readable but **not grown** — what is read is more than what the search
  need make (`Library.read_only`); growing them cost rung 2's evens filter.
- **Folds by induction** (`forms._induced`): examples whose lists differ by
  one element at the end give rows of the step's own spec; the step is
  tried over the loop's own scope first (the outer inputs change with it —
  `acc * n` fits the rows of a factorial and is not its step). Induction is
  the dearest subgoal and runs last, after the meet and repair.

Reading stays exact (`needs_exact.py`): every verified program that reads
passes its own tests from its tree — MBPP 88 → **123**/257, HumanEval 33
→ **55**/99.

| held, once (budget 8,000, meet + repair + forms) | solved / general |
|---|---|
| rung 1 | 28 / 21 (was 28 / 21) |
| rung 2 | 22 / 18 (was 21 / 19) |
| rung 3 | **21 / 17** |

## Rung 4 — editing a program that exists (`editing.py`, `bugs.py`)

The program is read with where every node came from (`Expr.where`: its
span, its operator's or member's span, a name as written). An **edit**
replaces one span's text, so it is made in the source — comments and
layout untouched — and the edited file is what is run. The edits are
general (operators and members with the same needs and gives, constants
nudged or to small numbers or others the program says, any name the
function declares of that type, sides swapped, unwrap, wrap), ranked by
suspicion (a node whose values on the failing cases are never its values
on the passing ones), one edit, then two. What no few edits fix — logic
missing — is **searched with the program as the sketch** (its parts in the
forward trie, itself a near miss) and the function written again
(`rewritten`, reported apart from edits).

Benchmarks: verified programs made wrong by one edit, **frozen**
(`bugs-dev.jsonl` from MBPP, `bugs-held.jsonl` from HumanEval — the edits
grew, the benchmark did not move with them); **HumanEvalFix** (164
human-written bugs, JavaScript, typed by MultiPL-E's signature: 157; 79
read). Judged by the bug's own tests (both test files for HumanEvalFix);
against the search solving the same cases from scratch.

| | fixed by edits | rewritten | unread | from scratch |
|---|---|---|---|---|
| generated, dev (40) | 30 (all one edit) | 7 | 0 | 21 |
| generated, held (40) | **36** (one edit) | 1 | 0 | 15 |
| **HumanEvalFix, held (157)** | **46** (43 one, 3 two) | **11** | 78 | 28 |

HumanEvalFix: **57 of 157** fixed, 57 of the 79 that read (72%); the
search from scratch passes 28. The 78 unread are the ceiling now —
objects and Sets, `sort` in place, types beyond the library's. As in
HumanEvalFix itself, the tests are given; a fix meets them, it is not a
claim about unseen cases.

**HumanEval-TS** (held, once, everything on: the reading's promise, the
whole-function decoder, rung 3's library and reader): 22 → **24**/151 pass
their tests; examples met 86 → 79, candidates 1.19M → 1.75M. More is
tried per task (induction last, a larger library) and more of what meets
the examples is right.

### The reader widened (2026-09-29)

What stopped reading was, underneath, a handful of general things, each
fixed as one:

- **The library was a gate on reading.** Anything the compiler resolves —
  a member, a global function (`isNaN`, `Number`), an operator on mixed
  types (`acc + (x === y)`), an index into anything — is read as an
  operator made there with the compiler's types when the library has none
  (`parse._op(made=...)`): it prints the same code, so it runs the same.
  The search does not grow them; what is read is more than what is grown.
- **What the tree does not model is opaque, not unread**: an object or a
  list written out, a regular expression, `new Set(...)`, a template, a
  callback whose body cannot be read — kept as its own text, every
  variable it reads from outside it a hole filled from the tree, printed
  in parentheses (an object after `=>` is otherwise a block — found by the
  exactness check).
- **Change in place is a change to a copy**: `c[k] = v`, `s.add(x)`,
  `l.sort(cmp)` on a local container read as the call made on a copy, the
  copy as it leaves it — the container's value before stays what it was.
- **Helpers as people write them**: typed by what they are given (a
  JavaScript parameter has no type of its own), made inside the function
  (inlined where called), calling themselves (a call to the function being
  read); an empty or undeclared value (`let x;`) is `undefined`; a global
  used as a value (`String`, `NaN`) is its name.

- **Values are shared, not copied.** Executing a body symbolically copies
  a variable's value wherever it is read; a program whose values are built
  from values built from values (`minPath`: guards reading what guards
  set) grew exponentially — one HumanEvalFix run held 40 GB and stalled.
  Now a large value (over 12 nodes) read a second time is **bound once
  where it was made** — the function's body, a loop's, a callback's
  (`enterFrame`/`leaveFrame`: the innermost that did not inherit it) —
  and every read refers to the binding; the same for a value kept as the
  other side of a merged `if`. The binding is **lazy and remembers**
  (`((b) => …)(thunk)`, read as `b()`): worked out when first read, once —
  eagerly, a value a guard kept from being worked out (`grid[m - 1]` at
  `m = 0`) would be. A program is read in about the size it was written
  (largest HumanEvalFix tree: 12k characters); twenty guards building on
  each other read and run as written (test).

Reading stays exact (`needs_exact.py`): **212/257** MBPP and **88/99**
HumanEval verified programs read and pass their tests from the tree (were
123 and 55); HumanEvalFix bugs that read: 79 → **131**/157.

**Budgets that mean what they say** (found by the runs, each a stall):
a subgoal's child search spends the parent's remaining budget, not 6,000
of its own (`forms._solve_hole`); the search's repair, which built every
subtree-times-kept-expression candidate of a near miss at once (285,000
full programs for a 1,900-node program read from a file), makes them as
checked and has its own declared allowance (`REPAIRS` = 20,000);
induction, which runs after the budget is spent, has one too
(`INDUCTIONS` = 12,000); a candidate that times out on one case is failed
without running the rest; a reported value past 100k characters is an
error, JSON's losses kept (`$set`, `$map`, functions refused). Dev rungs
1–3 unchanged (30/26, 22/20, 17/14) at fewer candidates.

**Rung 4, held, once, after widening** (the measured phase):

| | fixed by edits | rewritten | unread | from scratch |
|---|---|---|---|---|
| generated, held (40) | 36 (one edit) | 1 | 0 | 15 |
| **HumanEvalFix (157)** | **64** (61 one, 3 two) | **17** | 26 | 28 |

HumanEvalFix: **81 of 157** fixed (was 57), 81 of the 131 that read.
Rungs 1–3 held: 28/21, 22/18, **22/18** (rung 3 was 21/17).
HumanEval-TS (everything on): 24 → **25**/151 pass their tests, examples
met 79 → 82, candidates 1.75M → 1.17M, time 2,912 s → 1,333 s.

## Rung 5 — projects (`changing.py`, `projects.py`)

- **5a. A project read whole.** The compiler over several files at once
  (`tscheck.js` `programOfFiles`, imports resolved); a project's functions
  keyed `file#name`, a call through an import read as the function it
  names, an operator carrying its body; spans say which file. A project is
  run as its files are (`project`: each file a CommonJS module, `require`
  finding the project's own) — **inside the sandbox, under its timeout**:
  run by the checker directly, a helper made into an endless loop hung it
  (found when a benchmark build sat for 90 minutes). The compiler's errors
  come with their places (`diagnose`).
- **5b. The project's functions are the library**: every function the
  other files declare is an operator the search grows (`Spec.library`),
  read, not written; the answer written into the function's body, the
  project's tests judging.
- **5c. A fault in another file**: rung 4's edits over every function a
  test reaches, each edit made in its own file (`Edit.file`); `main` run
  with the project flattened to check, the project's tests to judge.
- **5d. A change is a plan; the compiler's errors are its impasses.** The
  first error is the top of the goal stack; the moves at its place are the
  name it cannot find as a name the project exports — or that export under
  the old name, at an import — and the arguments of the call it is in
  reordered; a move is kept when the errors fall; when none are left, the
  tests, and 5c for what still fails.

Benchmark (`projects.py`, **frozen**, built before measuring): projects of
2–3 modules assembled from verified programs (MBPP for dev, HumanEval for
held) — `use` (a `main` whose answer composes two of them), `bug` (`main`
as written, a helper made wrong by one edit), `change` (a helper renamed,
or its two parameters swapped, `main` and a second function left calling
it the old way). Every case is what the programs did on their own tests.

| held, once (15 each) | rung 5 | baseline |
|---|---|---|
| use | **10** | 5 — the search without the project's functions |
| bug | **13**, one edit each, in the faulty file | 5 — the search writing `main` again, around the fault |
| change | **15** — 10 renames in one edit (the export under the old name), 5 swaps in two (one per caller), 20 impasses met | 0 — rung 4's repair, which has no new names and no compiler |

Dev: use 11 (3), bug 11 (7), change 14 (0).

Honest limits: the projects are small and the changes are two kinds —
renames and swapped parameters, where the compiler says exactly where; a
change whose breakage the types cannot see (a parameter's meaning
changed, not its type) is left to 5c, which only has the tests.

## After rung 5: the two weakest numbers (2026-09-30)

### Every HumanEvalFix bug reads (26 did not)

What stopped the 26 was JavaScript as people write it, and wrong programs
being wrong. Each cause was one general thing:

- **JavaScript read as TypeScript** (`tscheck.js` `completed`). What the
  source leaves unsaid is said for it, from how it is used, *in the text the
  compiler is given*: a helper's untyped parameter takes the type of what
  it is first called with; a name assigned and never declared is declared
  where its function begins; a list begun empty that the compiler never
  settles (its elements set from its own) holds what is first pushed into
  it, or what the function says it returns. Only text is added, and
  `origin` maps every character back, so every span — and so every edit —
  is still of the source as written. (Before, the compiler's types inside
  such a helper were all `any`, and nothing in it resolved.)
- **A function returns whether or not it says so.** Every function begins
  not-returned (`DONE`, `RESULT`, as loops already had), callbacks and
  helpers too; falling off the end returns `undefined` unless it had
  returned; a guard inside a block whose rest goes on past it carries its
  return (`returned`); a bare `return` returns nothing.
- **A name nothing declares**, or a `let` read before its line, is its own
  text: it fails in the tree as it failed in the source.
- **What the language has had since** (`xs.at(-1)`): read with the
  compiler's newer library; the library the search grows from is unchanged.
- **Loops**: `for (k in o)` goes over `Object.keys(o)`; a test that is not
  a truth (`while (n)`) is one by `!!`; a loop that changes nothing outside
  itself is nothing in what the function returns.
- **Changes**: `c[j][k] = v` at any depth (each container on the way a copy
  with its own element set), `c[k]++`; a statement said for nothing
  (`xs.filter(f);`) is dropped. One notion of what code changes (`writes`:
  the name at the root of everything assigned, counted, deleted from or
  changed in place) serves all of it.

**Reading is checked on wrong programs too** (`needs_exact.py`): a bug's
tree, printed back, must give on the bug's cases what its source gives — the
same value, or fail where it fails. That check found five things that had
been read *silently wrong*, some of them since rung 3:

- `==` was read as `===`: `x == null` is true of what is undefined. Now
  only between two of one plain kind.
- **Two names, one list** (`let p = arr`, then `p[j] = …`, `return arr`):
  read as one name when either is changed in place; refused if either is
  then given another value.
- **An `if` whose two ways changed a name alike lost the change** (the
  merge kept the value from before the `if`).
- **A loop's state was worked out once for every place of it read** (a
  loop carrying three values ran three times, nested ones exponentially):
  bound once. **Work done before a loop and read inside it** was done
  every time round: bound where it was made, at its first read from inside.
- Running a program stops at the first case that runs out of time (a
  repair of a loop that never ends cost every case its timeout, for each
  of 3,000 edits).

| | before | now |
|---|---|---|
| HumanEvalFix bugs that read | 131 / 157 | **157 / 157**, all doing what their source does |
| verified programs read exactly | 212 / 257 MBPP, 88 / 99 HumanEval | **222** / 257, **92** / 99 |
| HumanEvalFix fixed (held, once) | 81 (64 edits, 17 rewritten) | **96** (76 edits — 73 one, 3 two — and 20 rewritten) |

Of the 26: 10 fixed by edits, 5 rewritten, 11 read and not fixed. Two that
were rewritten before are not now (the search wrote them; their trees
changed). Generated bugs, held: 36 + 1 of 40, unchanged. Rung 5 dev:
unchanged (11, 11, 14). Rungs 1–3 dev: the same to the candidate as the
commit before.

### A few examples are not the judge (HumanEval-TS: 25 of 151)

Diagnosed on MBPP dev (83 tasks the decoder was not taught from; one
example shown, as MBPP's prompts have none): 57 "solved", **18** right. Of
the wrong ones most were a constant or a parameter — `false`, `2`, `n`. Two
causes, both in how the search stops:

1. **The forward trie keeps one expression per vector of values on the
   examples.** With one example, `false` *is* every boolean expression
   that gives false there: nothing else with that value is kept, grown or
   offered.
2. **The first thing that meets the examples was the answer**, and the
   values the search begins with are tried first — before the decoder's
   proposals. On 8 tasks the decoder had written a right program and a
   constant was returned.

So, where there is a reading of the request (**judged**, `search.py`):

- two expressions are one only if they also do the same **beyond the
  examples** (on `meaning.probes`, inputs varied from them);
- what meets the examples is **kept with where it came from** and the
  answer chosen by that (`RANK`): a program the decoder wrote for this
  request; then one of its near misses **repaired by rung 4's edits**, the
  examples its cases (only with two or more examples); then what the search
  found — a part of a proposal, or grown; last a value it began with.
  Reading the decoder's work is not stopped part-way; after it, the search
  stops at the first program it finds.

Without a reading (rungs 1–3, rung 4's rewriting) nothing changed.

| MBPP dev, 83 tasks, same proposals | first that meets | judged |
|---|---|---|
| one example shown | 18 | **25** |
| two shown | 29 | **34** |

(With two shown, of 10 near misses edited to meet the examples 4 are right
— two the search had no answer or a wrong one for — and one displaces a
wrong program that happened to pass the tests.)

**Measured on dev and left out** — each chose no better than "the first":
ranking what meets the examples by how likely the reader of meaning finds
what it is made of, by its behaviour beyond the examples, by whether it
reads every parameter; looking further past the first program found (up to
the whole budget); taking the behaviour most of the decoder's samples agree
on; asking the decoder for 24 programs, not 8 (a right one on 26 tasks
either way); edits to near misses with one example (1 of 12 right). The
reader's `uses` order the search; they do not tell a right program from a
wrong one that meets the examples.

**HumanEval-TS, held — it did not carry.** Run once judged, then with each
part switched off, the same (seeded) proposals:

| HumanEval-TS, 151 tasks | pass their tests | meet the examples |
|---|---|---|
| the first that meets the examples (as it was) | 24 | 82 |
| judged, near misses not repaired | 24 | 81 |
| **judged** | **24** | 78 |

(The 25 before was another draw of the decoder's samples; sampling is
seeded now, `sketcher.SEED`, so a run can be run again.) Why dev's gain is
not there: HumanEval's prompts show two or more varied examples for most
tasks, so a constant seldom met them, and the decoder's right programs were
already being returned — 18 of the 24 are whole proposals in every variant.
Judging changes which *wrong* program is returned. By route, judged: whole
proposals 17 right of 23; near misses repaired 2 of 13 (dev: 4 of 10); what
the search finds from the examples alone 5 of 42.

So the number is the decoder's: a right program is returned where it wrote
one, and it writes one that meets the examples for 23 of 151 tasks.

### The decoder had been taught the search's dialect

`sketcher-functions` was taught at rung 3 from what the reader read then:
127 MBPP functions beside 4,000 generated programs in the search's own
dialect. The reader now reads nearly all of what people write — so the
decoder need not write the dialect. Two ways to use that, measured on dev
(the decoder alone: tasks where one of its programs is right / where the
first that meets the examples is):

| proposer | one example | two |
|---|---|---|
| `sketcher-functions` (as it was) | 26 / 24 | 26 / 26 |
| `sketcher-functions2`: taught again from what the reader reads now (234 MBPP functions, `sketches-functions2.jsonl`) | 34 / 31 | 33 / 33 |
| SmolLM2-360M-Instruct as it came, taught nothing | 35 / 33 | 37 / 36 |

Teaching the base model the dialect had cost more than it gave: its loss
on MBPP dev functions rises as it is taught (0.37 before, 0.56 after two
epochs). End to end, judged, on dev: `sketcher-functions2` 34 / **40**, the
base model 34 / 38 — so `sketcher-functions2` is the proposer
(`sketcher.PROPOSER`, `--sketcher proposer`; Adrian's choice after the held
runs below, 2026-10-01; a model directory with no `sketcher.json` is used
as one taught nothing).

**HumanEval-TS, held, once each: 24 → 30** with `sketcher-functions2`
(proposed 36, 23 right; near misses repaired 15, 4 right; the search's own
finds 3 of 33). With the base model, taught nothing (run once, not chosen):
**38** (proposed 45, 30 right; repaired near misses 20, 3 right; the
search's own 5 of 34).

Dev could not tell those two proposers apart; held can, by 8. Either the
base model is better at HumanEval's long docstring requests than one taught
on MBPP's one-line ones, or it has seen HumanEval: it was trained on public
code, HumanEval is the most copied benchmark there is, and MultiPL-E's
TypeScript translations are public too. Nothing here can say which, so 38
is not a claim about the architecture; 30 is the number chosen on dev. What
either writes is only ever a candidate — read into a tree, checked on the
examples, judged by the tests.

### Toward a majority, with 360M proposers (2026-10-01)

Adrian: get a majority a good solution, staying with 360M models (no larger
model at run time). Everything above says the number is the proposers', so
the work went there, measured on MBPP dev (two examples shown), the decoder
alone (the first of its programs that meets the examples is right):

| proposer | 8 programs | 32 |
|---|---|---|
| `sketcher-functions2` | 35 | 38 — its samples barely differ |
| SmolLM2-360M-Instruct, untaught | 33 | 43 |
| `sketcher-people` (below) | 34 | 40 |
| all three, 8 each | **47** | |

- **Asked as people ask** (the chat form) beats the request as a TypeScript
  docstring to complete (15 of 83 at 8 programs).
- **Asked again where nothing fits** (`proposals(rounds=...)`): what is easy
  is written once, what is hard is tried more.
- **Several proposers, each after the last** (`--sketcher a,b,c`): one is
  asked only where those before it wrote nothing that meets the examples.
  They are wrong in different places.
- **`sketcher-people`**: the base model taught, with no meaning line, whole
  functions as written — MBPP's verified solutions and **requests the teacher
  wrote** (`teach_requests.py`: SmolLM3 offline writes a new request in
  MultiPL-E's form from three MBPP train requests, then solves it; kept when
  its solution meets the request's own examples and reads; nothing sharing a
  name or eight words with HumanEval or MBPP dev). The teacher's run stopped
  at its time limit: 610 requests, 256 kept.

End to end on dev: **52** of 83 (was 34). Held, once (`--sketcher
proposers --rounds 2`): **HumanEval-TS 30 → 55 / 151** (36%) — proposed 73
(47 right), the search's own and repaired near misses 8 more. One run before
it stopped on a proposal whose printed helpers did not compile; helpers that
do not run are now a candidate failing (`tscheck.js` `values`), and such a
proposal is not offered (`sketcher._runs`).

Not a majority. 42 tasks get nothing that meets their examples; 26 get a
proposal that meets them and is wrong. The base model's share of the 55
carries the caveat above.

## Risk first: six estimators and the resolution matrix (2026-10-02/03)

Adrian's sprint deck, applied to a request (`PLAN.md`, "Risk first"):
six factors scored 0–3, high = 2 or more, a move per high factor and per
pair of them. Built: `risk.py` (labels, estimators, matrix), `tscheck.js`
`qualities`, `meaning.edge_probes` / `pair_probes`, the moves in
`search.py` (`_beyond`, `_settled`, `_chosen`, `solve`) and
`sketcher.proposals`; `experiment.py --risk estimators|oracle --moves
--dev --cache --part`.

**Labels** (`data/code-meaning/risk.jsonl`): D, P, S, X, B read off 4,744
verified programs; U from the untaught base model's 17 programs per request
on 584 requests (MBPP, the teacher's, HumanEval) — the run stopped at its
time limit with 28 of the teacher's requests unread (`label` resumes).
Labelled risk ranks failure (MBPP dev, the search as it was): passes by high
factors 0: 4/4, 1: 20/24, 2: 7/10, 3: 6/9, 4: 3/4.

**The six estimators** (`llm/risk-estimators`, heads over
`meaning-unixcoder` read only): high-vs-not, balanced (0.5 is chance):

| | D | P | S | U | X | B |
|---|---|---|---|---|---|---|
| MBPP dev (chosen on) | 0.86 | 0.73 | 0.72 | — (none high) | 0.76 | 0.79 |
| HumanEval held, once | **0.80** | 0.48 | 0.57 | 0.54 | 0.58 | 0.60 |

Depth carries; the rest barely do — chosen on 51 dev requests, and
HumanEval's docstrings are not MBPP's one-liners.

**Proposals made reproducible.** Each request is now sampled with a seed of
its own and the round's (`sketcher.proposals`): in a shared batch, asking
one request again (a matrix move) changed every other's samples, and a first
comparison moved ±5 tasks the matrix never touched. Per request it is about
5× slower to write, so what was written is kept (`--cache`): a run measured
again asks nothing it asked before. The baseline is unchanged by it: 52/83.

**MBPP dev, end to end** (same proposals, `meet+repair+forms+proposals`,
three 360M proposers, rounds 2):

| risk from | passes | evaluated | confirmed right | unconfirmed right |
|---|---|---|---|---|
| none (as before) | 52 | 210k | 19/20 | 33/44 |
| labels (oracle, 51 of 83) | **53** | 235k | 19/21 | 34/45 |
| estimators | 52 | 426k | 22/24 | 30/41 |
| estimators, no `budget` move | 52 | 260k | 22/24 | 30/40 |

- The matrix's moves barely change what is solved: one task more with the
  labels (`surface_Area`, from a proposal's part), none with the estimators,
  which put far more requests in shaded corners (76 of 83 vs 47) and so
  doubled the cost through the budget move for nothing. Dropped from the
  held run.
- What the risk assessment buys is knowing what to trust. **Confirmed**
  answers (a second program, written or found apart, does the same beyond
  the examples) are right 92–100% of the time; unconfirmed ones 73–77%.
  And the estimators' scores rank failure over all 83 requests: passes by
  high factors 0: 5/7, 1: 17/22, 2: 12/20, 3: 13/20, 4: 5/12, 5: 0/2.

**HumanEval-TS held, once** (in two halves, `--part`, the proposals kept;
the matrix without `budget`):

| | passes | evaluated | confirmed right | unconfirmed right |
|---|---|---|---|---|
| as before (per-request seeds) | 51 / 151 | 657k | 10/16 | 41/92 |
| risk first (estimators) | **54** / 151 | 757k | 14/25 | 40/86 |

- +4 −1 (`circular_shift`, `sort_array`, `skjkasdkd`, `choose_num` gained;
  `largest_prime_factor` lost to the search's own program chosen by
  agreement), for 15% more candidates. The old 55 was with batch-shared
  samples; with each request's own seed the same pipeline gives 51 —
  sampling, not the search.
- Risk ranks failure on held too: passes by estimated high factors 0: 3/8,
  1: 23/46, 2: 14/55, 3: 11/31, 4+: 3/11.
- Confirmation is a weaker signal on HumanEval than on MBPP (56% vs 47%
  right; the second half 3/10): two 360M programs agreeing beyond a few
  examples is weaker evidence where the requests are harder.

## What is next

The ladder's five rungs are built. What the numbers say is weakest is not a
rung but the writer, and what it is taught: HumanEval-TS moved when the
decoder was taught what people write rather than the search's dialect, and
not when the search chose better among what met the examples. A proposer
taught from more requests than MBPP's (its own verified programs, read back
through the exact reader, are the obvious corpus), and a judge that knows
more than where a program came from — which, measured here, the reader of
meaning as it stands is not.
