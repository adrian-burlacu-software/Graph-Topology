# v698: a VS Code harness — the conversation in the editor, projects read whole

Adrian (2026-10-03): "a harness for the ability to integrate into vscode and
read entire projects ... the url/port as a setting but with the 8697 default
so we can run any server under the harness and even restart the server
without restarting the harness."

## The shape

- **The extension is a thin client** (`tools/vscode-graph-topology/`, plain
  JavaScript, no build step, no dependencies). Everything it knows comes
  from the server over HTTP, so a server restart is a reconnect.
- **Settings**: `graphTopology.serverUrl` (default `http://127.0.0.1:8697`);
  `graphTopology.serverCommand` (empty: attach only; else the extension may
  start, stop and restart it, its output in a channel); the files a project
  is read from (glob, limits).
- **A protocol any server can speak** (`research/v698/protocol.md`):
  `GET /api/health` → name, version, an id per start (`instance`), and what
  it can do (`capabilities`). A changed instance is a restart: the extension
  sends the project again. What a server cannot do is greyed out, not
  broken.
- **What the extension does**:
  - a status-bar item (which server, connected or not; click for actions);
  - the conversation in a panel: the server's own page (v697's: everything
    a turn did, code blocks), in the editor's conversation
    (`?sid=vscode-<workspace>`);
  - *Read the project*: the workspace's files (glob, `.gitignore`, size
    caps) sent whole; each save sent again;
  - *Explain the selection*, *Write a function from the selection* (a
    comment, a signature, examples), *Ask about the project*: each a turn of
    the same conversation, the panel brought up to date;
  - an answer's code is offered, never written silently: insert, replace
    the selection, or open beside, the user choosing.
- **What the server adds** (`research/v698`, over v697's server):
  - `POST /api/say` (a request for code is longer than a URL),
    `POST /api/project`, `POST /api/file`, `GET /api/project`;
  - a project per conversation, read by the compiler: an **outline**
    (`tscheck.js` `outline`: each file's functions with their lines and
    types, what it imports, who calls whom) and **diagnostics**;
  - **the project in the conversation** (an act, as code is): what is in
    it, where a function is, what calls it, what it calls, what it does
    (read exactly: the compiler's structure), what is wrong with it
    (the compiler's errors) — and code pasted with a question about it
    (*explain the selection*) read the same way and kept as the
    conversation's code, so *what does it do*, a call, *what else* follow.
  - the everything view gains the project: the call graph drawn.

## Code talk read by the encoder (Adrian, 2026-10-03: "natural language is read by an encoder, never routing tables")

The structure stays (`asking.py`: a subject and an aspect, gaps); what
reads text into it becomes the shared reader (`research/encoder.py`), taught
code talk as a subject of its own beside mathematics and design:

    code_act      none | ask | make | change | run | teach
    code_aspect   none | explain | where | size | callers | calls | bugs |
                  sure | why | others | risk | files | functions
    code_subject  none | project | file | named | last
    code_role     each word: the subject's phrase (SUBJ), or -- teaching --
                  the concept (CONCEPT) and its members (MEMBER)

- **Taught offline** (`teach_code_talk.py`): SmolLM3 writes messages for
  each act, aspect and kind of subject with a placeholder where the subject
  is named; the placeholder is filled with real names (MBPP / HumanEval
  entries, v696's projects' files, project names the teacher lists) — the
  labels known by construction. Real requests (MBPP's English) are `make`,
  real examples (`name(args) == value`) `change`. Not code talk: v689's
  reader corpus, and messages the teacher writes using *bug*, *error*,
  *file*, *function*, *run* in their everyday senses.
- **The reader**: `teach_reader train --subject math --subject design
  --subject code` into a new `llm/reader-code`, chosen as the shared
  reader only if the suite and v689's quick evaluation hold.
- **At run time**: the encoder reads; what a SUBJ span names is resolved
  exactly (the project's index, its files, the conversation's code); `last`
  is whatever was talked about last. Below a floor of how sure, the turn is
  not taken. The hand rules of `asking.route` are left only as a check.
- **Teaching knowledge** (*a vowel is one of a, e, i, o, u*): the `teach`
  act, its CONCEPT and MEMBER spans, kept long-term and used in writing code
  (next).
- Measured on held-out phrasings and names, against the hand rules.

## Measured / checked

Server endpoints and the project act over HTTP, on a small project and on
this repository's own TypeScript (`research/v696/tscheck.js` is JavaScript;
the v696 projects' files). `test_v698.py`. The extension: `node --check`,
its pure parts unit-run under Node with a stub of `vscode`; installed into
VS Code and driven by hand (reported as such: no headless VS Code here).

## Rules held to

As before: commit only when asked; nothing written into the user's files
without their choosing; models untouched; the broader fix.
