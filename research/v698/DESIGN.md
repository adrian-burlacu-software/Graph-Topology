# v698: a VS Code harness — what was built

`PLAN.md` is the contract; `protocol.md` what a server speaks to be
attached. Branch `v698/vscode-harness`, from master after v697 (`239dd36`).

## The server (`research/v698`, over v697's)

    python -m research.v698 --workers 19        port 8697

- `GET /api/health`: name, server, git version, an `instance` new at every
  start, capabilities. `POST /api/say` (any length), `POST /api/project`,
  `POST /api/file`, `GET /api/project` (summary, outline, call graph).
- **A project per conversation** (`project.py`): files under one root for
  the checker; **the outline** (`tscheck.js` `outline`, syntax only — a
  file that does not compile is still outlined): functions declared, bound
  to a name, class methods, with lines, parameters and result as written,
  what each calls by name; each file's imports. **Calls resolved** across
  files (own functions, then relative imports, `index` files); **the
  compiler's diagnostics**, with lines, on demand.
- **The project in the conversation** (`asking.py`, `page.py`; utility 211,
  above v697's code at 210): overview, files, a file's functions, where,
  who calls, what it calls (the project's, then the language's), what a
  function is (signature, lines, the compiler's structure, callers), what
  the compiler finds wrong. A new step, `looked`.
- **Pasted code** (utility 212): code in a fence with *explain* / *what
  does* … is read (`outline`, `structure`, run on inputs of its types, the
  compiler alone on it) and becomes the conversation's code, so a call,
  *what does it do*, *how sure* follow it.
- v697's server took a `STEPS` hook and a `main(handler, conversations)`
  so v698 reuses it whole.

## The extension (`tools/vscode-graph-topology`)

Plain JavaScript, no dependencies, no build; `package.py` packs the `.vsix`
with the standard library (`--install` runs `code --install-extension`);
`regenerate` step `vscode-extension`.

- **Attach or run.** `serverUrl` (default `http://127.0.0.1:8697`) is
  asked `/api/health` every 3 s; the status bar says which server, and
  whether the project is read. `serverCommand` (+ `serverCwd`,
  `autoStart`): start / stop / restart from the editor (Windows: the whole
  tree, `taskkill /T`), its output in a channel. A changed `instance` — a
  restart, by the editor or anyone — sends the project again.
- **The conversation** in a panel: the server's own page framed, this
  workspace's conversation (`vscode-<hash of the folder>`). The page knows
  it is framed: code blocks gain *open* (a project function where it is)
  and *insert* (the editor asks: at the cursor, replace the selection,
  below it, beside as a new file, copy). Nothing is written without that
  choice.
- **Commands**: read the project (glob, excludes, file and size caps; each
  save and deletion sent after), ask, explain the selection, write a
  function from the selection (comment markers taken off; the answer
  offered for the selection), start / stop / restart, output, settings —
  all from the status bar.

## Checked

- `test_v698.py` (5): outline, calls across files (an import, an arrow
  function, a class method), diagnostics with lines, a change and a
  deletion, every question kind and what is not one (*where is Mary*,
  *can it swim*), pasted code read, kept and called; the extension parses
  and its library's tests pass (`test.js`, under Node).
- Over HTTP, as the editor drives it: health, a project of five files (one
  broken: the compiler's error, by line), the questions, pasted code then a
  call and *what does it do*, *is a dog an animal* still v689's; a restart
  gives a new instance.
- The `.vsix` installs (`code --list-extensions`). **Not driven inside VS
  Code here** (no headless editor): the panel, the commands and the restart
  path in the editor are untested by me.
- The full suite: 241 of 242; the one failure is master's
  (`v689.test_time…claim_said_as_a_question_is_confirmed`).
