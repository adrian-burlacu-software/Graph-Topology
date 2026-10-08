# v700: the attachment — changes asked of the project, made in its files

Claude attached to the architecture through an MCP (`tools/graph-topology-mcp`)
and asked it, in the dense sentences people use to describe a fix, to change
the architecture's own code. Where it could not do what Claude would have
done, the architecture was changed — not the request.

## What it could not do (2026-10-07, first statements)

| said | read as | done |
|---|---|---|
| `In tools/graph-topology-mcp/server.py, _turn reads the turn's number from "n" or "turn", but the server sends it as "number" -- read "number".` | ask / explain, of the file | described the file |
| `fix _turn: it should read the turn number from "number"` | change, of the last answer | rewrote an unrelated function of the conversation |
| `change _fence so that it returns "py" for Python files` | not code talk | "I haven't been told anything about it." |
| `_relative crashes when the path is on another drive; return the path as given` | ask / bugs | "Nothing contradicts them." |

Three gaps, all of the architecture:

1. **Reading.** A change of the project's code was never taught: `change`
   meant the conversation's last answer, and the dense statement of a fault
   (`X reads ... but ...`) was nothing it had read.
2. **Editing.** Nothing changes a function as asked. The writers (v696),
   given the statement and `_turn`, each wrote `_turn` back unchanged: they
   were taught to write a function from a request, not to change one.
3. **Writing.** Nothing writes a file. The server held the project in
   memory; the bridge put answers into files — the bridge doing the work.

## What was built

- **The editor** (`teach_editor.py`, `llm/editor`): SmolLM2-360M taught on
  CommitPackFT (bigcode; Python and TypeScript commits of permissively
  licensed repositories): the message, the path, the part of the file the
  change is in (the function, else the lines around it) → the change as
  `<<< old === new >>>` blocks. Blocks, not the function said again: an
  answer is put in where its old lines are found once, or refused.
- **The change made** (`fixing.py`): the part named (a function, or the
  lines of a file around the code names the statement mentions — a lookup
  in the code's own names, not a reading), the editor's answers (greedy +
  6 sampled), each checked — it parses, the compiler finds nothing more
  wrong — the most agreed taken and written: in the project, and on disk
  under the project's `root`. What was there is kept (`undo`).
- **The project on disk** (`v698/project.py`): `root`, `on_disk` (never
  outside it), `write`; `POST /api/project` takes `root`; capability
  `edit` (`protocol.md`).
- **The reading** (`v698/teach_code_talk.py` `commit_rows`): commit
  messages taught as `change` — of the function they name, where the name
  is written as code is (`load_data_frame`, `run()`; not `default`), of
  the file said first (`in setup.py, bump ...`), else of the code talked
  of. `v698/page.py`: a change whose subject is the project's function or
  file goes to `fixing.change`.
- **The bridge** no longer edits files: `say` carries the words; the reply
  carries the change made (`file`, `lines`, `diff`, `written`).

## Numbers

(to come: the editor on held commits; the reader on held commit messages
and v689's suite; the statements of the attachment, end to end.)
