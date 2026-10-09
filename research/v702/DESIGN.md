# v702: Bash, for basic computer use

Adrian (2026-10-09): "integrate writing and reading Bash commands and results
into the architecture. This is for basic computer use." Asked through the
MCP first, the architecture had no shell: `run ls research/v701` -- "I don't
know what ls research/v701 is"; `what files are in research/v701?` -- looked
for in the project's data; `$ git log --oneline -3` -- a TypeScript function.

## What it does now

A message to the computer is read by the shared reader (`reader-code30`,
heads `shell_act`: none | command | task | yes | no, and `shell_role`
marking a command's words) and carried out by the page's `shell` act
(before code and data talk):

- **a command given** (`run ls research/v701`, `$ git log -3`, `git
  status`) -- cut from the message as written (its case, its quoting);
- **a task** (`show me the last 3 commits`, `how big is the research
  folder?`) -- the command written by `llm/shell-writer2`: seven answers,
  those that parse (`bash -n`) and read no unset variable (`$DATA_DIR`),
  the one most write -- two at least; where none agree, commands that only
  read are run and those printing the same agree (`_agreed_by_output`);
- **yes / no** to a command shown and waiting.

**What runs without asking** (`shell.reads_only`): the command is parsed
into the programs it runs; each must be one that only reads (`ls`, `cat`,
`du`, `grep`, `git log/status/diff/show`, `find` without `-delete`/`-exec`,
`sed` without `-i` ...), nothing written to a file, nothing run inside it
(`$(...)`). Anything else is shown with why (`find is given -delete`) and
runs on yes -- or at once where the server is started `--no-ask`.

**Running** (`shell.run`): Git Bash, in the project's folder (its `root`),
under GNU `timeout` (Windows' taskkill stops Git Bash, not the MSYS programs
under it: a `find $HOME` outlived its run), output capped. What it printed
is said; where it is JSON or a table it is held as the conversation's data
(`v701`), and asked about next.

## What was taught

| model | from | taught |
|---|---|---|
| `shell-writer` | SmolLM2-360M | NL2Bash's 12,607 descriptions and commands, and 3,802 of them said again by the teacher as the same request |
| `shell-writer2` | shell-writer | commands made over this project's own files (`git log -n 3`, `du -sh research`, `wc -l regenerate.py`), the teacher saying what each asks |
| `reader-code29` / `30` | 28 / 29 | a command given, a task, yes, no; requests to change a program that talk of commands as none; words told as code now and then |

Writer, on requests: last 3 commits `git log -n 3` (6 of 7; was `git log -n
3 | tail -1`), current branch `git branch --show-current` (6; was unsure),
lines of a file `wc -l` (7). Reader: shell act 99.6% held; suite 1575 pass.

## Shortfalls met, and the fix of each class

- the teacher, told to ask "how many", changed what was asked: requests are
  said again as the same request, sampled, lines that are the command left
  out;
- told to write requests and commands both, it wrote `request ||| ...`:
  commands are made here, the teacher only says what each asks;
- the writer put the request's paths at the computer's root
  (`/research/v701`): the request's paths are kept as it names them;
- `the` names a function in this project: told so, a task fell under the
  floor -- tags are taught on any row, so a tag alone decides nothing;
- Adrian's own ask was written `brew install`: requests to change a program
  are taught as none, and read now as a change of the project;
- `git log`'s output was held as CSV: only JSON or rows of the same columns
  are held;
- `who calls shell.run` was written `who -m | grep`: a task code talk reads
  as of something the project has is code talk's.

## Left

- `how many python files are in research/v702` is written `ls ... | wc -l`
  (counts `__pycache__` too); `how much free disk space is there?` is read
  as no shell talk.
- Code talk does not find `shell.run` (a module's function by a dotted
  name).
- A request to change the architecture by a whole subsystem (Adrian's ask
  itself) is read as a change, and the editor, which changes one function,
  makes none -- v703.
