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

## Measured / checked

Server endpoints and the project act over HTTP, on a small project and on
this repository's own TypeScript (`research/v696/tscheck.js` is JavaScript;
the v696 projects' files). `test_v698.py`. The extension: `node --check`,
its pure parts unit-run under Node with a stub of `vscode`; installed into
VS Code and driven by hand (reported as such: no headless VS Code here).

## Rules held to

As before: commit only when asked; nothing written into the user's files
without their choosing; models untouched; the broader fix.
