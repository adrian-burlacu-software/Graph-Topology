# v706 roadmap: from a self-extending answerer to a self-improving coding agent

Agreed with Adrian on 2026-11-12, after the loop kept its first self-made
improvement (`a047f1d`: rental 31 to 35 of 36, clinic 24 to 27, core unchanged,
nothing lost, no model writing the change).

## Where it stands

- Answers are short typed queries over a model of the project (`queries.py`).
- A failed case is solved by enumerating the queries its words could mean and
  keeping the first that passes the case's check (`synth.py`, `learn.py`).
- Shared steps get a name and become vocabulary (`ways.abstract`).
- It writes its own questions with answers computed from the project
  (`curriculum.py`).
- `rsi.py` learns, measures SEEN, HELD and core, and commits or reverts.

## The rule for every step below

A step is done when the loop keeps it: `rental` gains and loses nothing,
`clinic` (never looked at) and core lose nothing. No model writes a change;
models only read the message and say the answer.

## Steps, in order

1. **Learn what words mean.** Today a query is learnable only when the
   message's words match the project (`classes`, `tests`). When a search with
   one unexplained word and one unexplained step passes the check (`dearest`
   and "highest value"), keep the pairing and teach it to the reader. Fixes
   the remaining probe failures. Days.

2. **Changes as programs.** An edit as a short typed program over the code
   model (rename, add a parameter, guard a case, change a constant, move a
   call), found by search and checked by tests, the way answers now are. The
   v705 pieces (`shaped`, `repair`, `splice`) are the primitives. This is the
   step that takes self-improvement from answering to writing code, and the
   one with the most risk: the space is far larger than queries.

3. **It builds its own world model.** The raises, calls and parameters
   collections were added by hand. It should derive collections itself, from
   syntax trees and from watching code run, and keep those that let it solve
   cases it could not before.

4. **Composing and inventing.** Queries feeding queries, recursion (callers
   of callers), a growing library of named parts, and "dreaming": composing
   its parts into questions and changes nobody asked for, keeping the ones
   that prove useful or compress what it knows.

5. **On itself.** Its own code is already one of the curriculum's projects.
   Once step 2 holds on small changes, the loop's own parts (the search's
   ranking, the keep rule's guards, the world model) become things it changes
   by the same rule.

## Known open items

- Three `list the classes in X.py` questions are read as shell requests by
  the reader; four `does f raise anything?` questions are answered with the
  exception and no "yes".
- `what group is it in?` needs the turn before it answered right first.
- A loop round with `--curriculum` has not been run.
- Answers are said by template, not by the decoder.
