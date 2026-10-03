# The harness protocol (v698)

What a server answers to be attached by the editor
(`tools/vscode-graph-topology`). Everything is JSON over HTTP on
`graphTopology.serverUrl` (default `http://127.0.0.1:8697`); an error is
`{"error": "..."}` with a 4xx/5xx status.

## Required

`GET /api/health`

    {"name": "graph-topology", "server": "v698", "version": "<git rev>",
     "instance": "<an id new at every start>", "started": <unix time>,
     "capabilities": ["say", "turn", "page", "code", "project", "file",
                      "pasted"], "page": "/", "projects": <n>}

The editor asks every 3 s. No answer: down. A different `instance`: a
restart — what the server held (a project) is gone, and is sent again.
A command whose capability is missing is refused by the editor before it is
sent.

`GET /` (`page`) — the conversation's page; `?sid=` opens a conversation.
A framed page may post `{source: "graph-topology", type: "open", file,
line}` or `{..., type: "insert", code}` to its parent; the editor opens
the file, or asks the user where the code goes.

## Capabilities

| capability | endpoint | body / query | reply |
|---|---|---|---|
| `say` | `POST /api/say` | `{sid, q, example?}` | the turn (v697's: `reply`, `answer`, `steps`, …) |
| `turn` | `GET /api/turn` | `?sid=&n=` | the turn whole, with `graph` |
| `project` | `POST /api/project` | `{sid, files: {path: text}, name}` | `{refused, name, files, lines, functions, exported, "most called"}` |
| `project` | `GET /api/project` | `?sid=` | `{summary, outline, graph}` |
| `file` | `POST /api/file` | `{sid, path, text \| null}` | as `POST /api/project` |
| `code`, `pasted` | (through `say`) | a request for code; code in a fence with a question | |

Paths are the workspace's, relative, `/`-separated. A conversation id from
the editor is `vscode-<12 hex of the workspace path>`, so a workspace
reopened is the same conversation.
