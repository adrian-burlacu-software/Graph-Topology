# Graph Topology for VS Code

The Graph-Topology conversation in the editor: everything each turn did,
code written and checked, projects read whole. A thin client: everything it
knows comes from a server at a URL you choose, so the server can be
restarted (or swapped) without restarting the editor.

## Install

    python tools/vscode-graph-topology/package.py --install

(or build the `.vsix` without `--install` and use *Extensions: Install from
VSIX…*). Then reload the window.

## Use

Start a server (`python -m research.v698 --workers 19`, port 8697), or let
the editor start it: set `graphTopology.serverCommand`, then *Graph
Topology: Start the Server* (or `graphTopology.autoStart`). The status bar
says which server is attached; click it for every action.

- **Open the Conversation** — the server's page beside the code, this
  workspace's own conversation. Code blocks there can be **opened** where
  they are or **inserted**: the editor asks where; nothing is written
  without that.
- **Read the Project** — the workspace's TypeScript and JavaScript sent
  whole; each save sent again. Then ask: *what is in the project*, *where
  is main*, *who calls parse*, *what does render do*, *does the project
  compile*.
- **Explain the Selection** — the selected code, read and run.
- **Write a Function from the Selection** — select a comment, a signature
  and examples (`total([1, 2]) == 3`); the answer is offered for the
  selection.
- **Ask…** — anything, in this workspace's conversation.
- **Start / Stop / Restart the Server** — when this editor runs it; its
  output is in the *Graph Topology* channel.

## Settings

| setting | default | |
|---|---|---|
| `graphTopology.serverUrl` | `http://127.0.0.1:8697` | any server that answers `/api/health` |
| `graphTopology.serverCommand` | empty | how to start it; empty: attach only |
| `graphTopology.serverCwd` | the workspace | where that runs |
| `graphTopology.autoStart` | false | start it when none answers |
| `graphTopology.projectInclude` | TS and JS | what is read |
| `graphTopology.projectExclude` | node_modules, dist, build, … | what never is |
| `graphTopology.maxFiles` / `maxFileBytes` | 1500 / 300 KB | limits |
| `graphTopology.syncOnSave` | true | send a file again when saved |

The protocol a server speaks to be attached: `research/v698/protocol.md`.
