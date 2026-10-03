# v697: transparency — everything a turn did, on the page, code included

Adrian (2026-10-03): "exposing EVERYTHING (IMPORTANT) for each conversation
turn (from semantic graph to searches, to results). Kind of like what we
have already with v689 but also include code." Then questions to improve
transparency or generality. Code blocks "extra easy to use … special
formatting inside their own little sections". Files and OPFS later.

## What exists (surveyed 2026-10-03)

- The page is v690's (`v690/server.py`, `app.html`, port 8690). A turn is
  v689's `Turn.as_dict`, **trimmed** (`v689.server.trimmed`: v688's run cut
  to its summary), plus `reply` and `steps` (`v690/steps.py`: heard, read,
  resolved, reasoned, planned, remembered, answered, said — each a line and
  a detail). The raw `trace` (operators per claim) and `executed` (every
  executive run, with its conflict sets) are on the turn but never shown.
- Later layers answer through `v689.session.contributes` (an executive
  operator; v692 does mathematics this way) and v691's hooks.
- v696 (code: reader of meaning, risk, proposers, search) is reachable from
  no conversation turn.
- Code on the page: a `.mono` class. No blocks, no copy.

## What is built

1. **Everything, kept.** The v697 server keeps each turn whole before it is
   trimmed — v688's whole run, every executive run, the walk — and serves it
   (`/api/turn?sid&n`). The page's **inspector** shows it: every step as
   now, then each part of the turn in its own panel — the semantic graph the
   turn touched (the walk's and the run's facts as nodes and edges, drawn),
   the executive runs (each cycle's conflict set, what fired, what it was
   chosen over), and the raw turn as a collapsible, searchable tree.
2. **Code as an act of the conversation.** `v697/coding.py` + an operator
   registered with `v689.session.contributes`: a request for a function —
   English, and optionally a TypeScript signature and examples
   (`reverse("ab") == "ba"`) — is read into a v696 `Spec` and solved by the
   whole v696 pipeline: the reader of meaning, the six risk estimators and
   the matrix's moves, the three 360M proposers (kept loaded, their writing
   cached), the judged search. The answer carries **everything**: the
   request as read, what the reader expects, the risks and their shaded
   corners and moves, every proposal as written and as read back (meets the
   examples or not, behaviour beyond them), the search's stages as they
   happened (`search.Solver.events`), every program that met the examples,
   how the answer was chosen and whether a second program confirmed it,
   what it cost — and the program printed whole. Without examples the
   proposals are run but nothing verifies them, and the turn says so.
3. **Code blocks** (`app.html`): each in its own section — a header with the
   language, a name and what it is (answer / proposal / near miss),
   line numbers outside the selection, highlighting, and buttons: copy,
   download as `.ts`, wrap, fold. Proposals as a stack of small blocks with
   chips (parsed, meets the examples, chosen, confirmed).
4. **A `programmed` step** in the account (`v697/steps.py`, over v690's):
   one line for what was done, the detail being the code panel.

## Rules held to

As v696: models in `llm/` untouchable, nothing new that is gitignored
without `regenerate`, commit only when asked, the broader fix. The v690
page stays at `/v690`; v697's is `/`.
