# v697: transparency — what was built

`PLAN.md` is the contract. Started 2026-10-03 on `v697/transparency`, from
master after v696 was merged (`2c42f89`).

    python -m research.v697 --workers 19      http://127.0.0.1:8697

## Everything a turn did

- **Kept whole.** `server.Conversations` copies each turn before v689's trim
  (v688's run whole, the walk, the reading, memory) and keeps the last 60
  per conversation in memory: `/api/turn?sid=&n=`. The archive is unchanged
  (trimmed); a turn from before the server started has its steps, not
  everything.
- **The executive's runs were never on a turn**: `Turn.executed` (every run,
  each cycle's conflict set) was filled and not put in `as_dict`. It is now;
  `v689.server.trimmed` leaves it out of the archive. A plain question
  ("can it swim") carries 39 runs.
- **The semantic graph** (`graph.py`): nodes and edges from the walk (the
  chain, the evidence with source and confidence, each node's rules), the
  conversation's individuals, what each phrase was resolved to, v688's
  activation and doubts — each saying where it came from. Drawn on the page
  by a small spring layout, with the edge table under it.
- **The inspector** (the page's *everything* tab; a turn's *everything*
  button): the summary, the steps, the code, the graph, the executive (each
  run: what fired, what it was chosen over, the rest of the run), the walk,
  v688's question loop (each cycle's questions and verdicts, its doubts, its
  working memory), the reading, memory, and the raw turn as a tree that
  opens, with a filter, copy and download.

## Code as an act of the conversation

- `coding.read`: English, and optionally a TypeScript signature and examples
  (`f(args) == value`, also `===`, `=>`, `->`, `returns`), one per line or
  several on one; a signature is made from the examples' types where none is
  given; without either it asks for an example. Mathematics (`let f(x) =
  …`, `what is f(3)`) is not taken.
- `page.py` registers `programming` with v689's executive (utility 210,
  above mathematics); it reads the utterance as typed (v689's reading
  normalises it).
- `coding.solve`: the whole of v696 — the reader of meaning, the six risk
  estimators (sharing its encoder) and the matrix's moves (without
  `budget`), the three writers (every one asked once: `readings`), the
  judged search — and the answer carries all of it: the request as read,
  the reading, the risks and corners, every program written per writer per
  round (read? meets?), the search's events (`search.Solver.events`: stages,
  levels, every program that met or was refused, how the answer was chosen
  and every behaviour beyond the examples with who arrived at it), timings.
- **Confirmed means another writer, not another text** (v696 changed with
  it): `Spec.authors` records which decoder wrote each program, and
  `search._authors` counts who arrived at a behaviour apart. Before, three
  writers writing the same program counted as one.

## Code in a conversation (2026-10-03, Adrian: "I was clarifying the code but it was not able to pick that up")

- **A workspace per conversation** (`conversation.py`): the last request as
  read and everything solving it came to. What is said next is classified:
  a new request; **more of it** (an example or signature of the same
  function, or *it should return "Hello World!"*), merged with what was
  asked and solved again; **a call** (`hello_world()`, *run it on [4, 5]*),
  run; **a question** — what it does, why that one, how sure, what else,
  its risks — answered from the record and by running, never made up. A
  question counts as about code only right after a code turn or where it
  names the code: *can it swim* after code is still v689's.
- **What it does** is read exactly: the compiler's structure (`uses`, the
  root) and `meaning.behaviour` over the examples and inputs varied from
  them — v696's teachers, not the learned reader.
- **What else**: the other behaviours that met the examples, and the first
  input where they part from the answer, offered as the question to settle
  (`vowels("AEIOU") == ...`).
- **In words alone** (*I want a program that prints out "Hello World"*):
  every writer writes for the request as said; programs are run — on the
  examples or on inputs of their types — and grouped by what they do; the
  answer is what the most writers arrived at apart (confirmed by two). A
  quoted string after *prints* / *returns* is an example from the words.
  Writers' functions without types are read (`void` where nothing is
  returned), and bare statements are a function of nothing (`main`).
- **Printing is behaviour**: `tscheck.js` `run` keeps what a call prints.
- **A function of nothing may be a constant**: the search set every
  constant aside as "what it began with" (a constant meets one example of
  anything) — but for a function with no parameters it is the program, and
  setting it aside made `"Hello World!".split("").map(...).join("")` the
  answer. Now `pool` only where the spec has parameters.

Replayed over HTTP: the screenshot's conversation (hello world → the
example → confirmed `return "Hello World!"` by two writers), words alone
(printHelloWorld, confirmed), vowels → *what else* → `vowels("AEIOU")`
gives 0 → `vowels("AEIOU") == 5` → re-solved, confirmed, gives 5.

## Code blocks

Each its own section (`codeBlock` in `app.html`): a header (language, name,
what it is, chips — confirmed / meets / route / as written), line numbers
outside the selection, TypeScript highlighting, copy (clipboard, with the old
fallback), download as `.ts`, wrap, fold; long code folds at 30 lines. The
answer sits under the reply; proposals stack as closed blocks with their
chips; the raw turn copies or downloads as JSON. What is typed is a textarea:
Enter says it, Shift+Enter a new line, and it turns monospace when it looks
like code.

## Measured

Over HTTP: an ordinary turn and three requests for code (sum, reverse
words, vowels) — the first code request ~90 s (the writers loaded, the
first writer asked three rounds), later ones seconds. `test_v697.py`: 8.
The full suite: 241 of 242; the one failure
(`v689.test_time…claim_said_as_a_question_is_confirmed`) fails on master
too.
