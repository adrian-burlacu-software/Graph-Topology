// Graph Topology in VS Code: a thin client of a server at a URL.
//
// Everything it knows comes from the server over HTTP, so a server restart
// is a reconnect: /api/health is asked every few seconds, and a changed
// `instance` is a new start -- the project is sent again. It may also run
// the server itself (graphTopology.serverCommand), and restart it without
// restarting the editor.
//
// Nothing it is given is written into a file without the user choosing
// where (insert at the cursor, replace the selection, open beside).
"use strict";

const vscode = require("vscode");
const childProcess = require("child_process");
const crypto = require("crypto");
const lib = require("./lib");

let output;            // the server's output, and the harness's own notes
let status;            // the status-bar item
let panel = null;      // the conversation, when open
let child = null;      // the server process, when this editor started it
let health = null;     // the last /api/health
let projectRead = false;
let timer = null;
let lastEditor = null; // the editor a code block is put into

function config() {
  return vscode.workspace.getConfiguration("graphTopology");
}
function serverUrl() {
  return lib.base(config().get("serverUrl"));
}
function folder() {
  const folders = vscode.workspace.workspaceFolders || [];
  return folders.length ? folders[0] : null;
}
function sid() {
  return lib.sidFor(folder() ? folder().uri.fsPath : "");
}
function note(text) {
  output.appendLine(`[harness ${new Date().toLocaleTimeString()}] ${text}`);
}

// -- talking to the server ----------------------------------------------------
async function request(path, body, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs || 15000);
  try {
    const response = await fetch(serverUrl() + path, body === undefined ? {
      signal: controller.signal } : {
      method: "POST", signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const found = await response.json();
    if (found && found.error) throw new Error(found.error);
    return found;
  } finally {
    clearTimeout(timeout);
  }
}

async function say(text) {
  if (!health) throw new Error(`no server at ${serverUrl()}`);
  return vscode.window.withProgress({
    location: vscode.ProgressLocation.Notification,
    title: "Graph Topology: reading, thinking, writing …", cancellable: false,
  }, async () => {
    // a request for code may take a minute and more: writers, then search
    const turn = await request("/api/say", { sid: sid(), q: text }, 15 * 60 * 1000);
    refreshPanel();
    return turn;
  });
}

// -- health: up, down, restarted -----------------------------------------------
async function check() {
  let found = null;
  try {
    found = await request("/api/health", undefined, 3000);
  } catch (_) {
    found = null;
  }
  const seen = lib.compare(health, found);
  const was = !!health;
  health = seen.up ? found : null;
  if (seen.up && (!was || seen.restarted)) {
    note(`server ${found.server || "?"} ${found.version || ""} up at ${serverUrl()}` +
         (seen.restarted ? " (restarted)" : ""));
    if (panel) panel.webview.postMessage({ type: "up" });
    refreshPanel();
    if (projectRead && (seen.restarted || !was)) {
      // what it held went with the old process: send it again
      readProject(true).catch((bad) => note(`project not sent again: ${bad.message}`));
    }
  } else if (!seen.up && was) {
    note(`server at ${serverUrl()} is not answering`);
    if (panel) panel.webview.postMessage({ type: "down" });
  }
  paint();
}

function paint() {
  if (health) {
    status.text = `$(pass-filled) GT ${health.server || ""}`;
    status.tooltip = `Graph Topology: ${health.server} ${health.version || ""} at ${serverUrl()}` +
      `\ncan: ${(health.capabilities || []).join(", ")}` +
      (projectRead ? "\nthe project is read" : "\nthe project is not read") +
      (child ? "\nstarted by this editor" : "") + "\n\nclick for actions";
    status.backgroundColor = undefined;
  } else {
    status.text = child ? "$(sync~spin) GT starting" : "$(circle-slash) GT";
    status.tooltip = `Graph Topology: no server at ${serverUrl()}\nclick for actions`;
    status.backgroundColor = new vscode.ThemeColor("statusBarItem.warningBackground");
  }
}

function can(capability) {
  return !!health && (health.capabilities || []).includes(capability);
}

// -- the server, when this editor runs it -----------------------------------------
function startServer() {
  const command = (config().get("serverCommand") || "").trim();
  if (!command) {
    vscode.window.showWarningMessage(
      "Graph Topology: no server command set (graphTopology.serverCommand): attach to a running server, or set one.",
      "Open settings").then((pick) => pick && vscode.commands.executeCommand(
        "workbench.action.openSettings", "graphTopology.serverCommand"));
    return;
  }
  if (child) {
    vscode.window.showInformationMessage("Graph Topology: the server is already running.");
    return;
  }
  const cwd = config().get("serverCwd") || (folder() ? folder().uri.fsPath : undefined);
  note(`starting: ${command} (in ${cwd || "the default folder"})`);
  output.show(true);
  child = childProcess.spawn(command, { cwd, shell: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" } });
  child.stdout.on("data", (data) => output.append(String(data)));
  child.stderr.on("data", (data) => output.append(String(data)));
  child.on("exit", (code, signal) => {
    note(`the server stopped (${signal || code})`);
    child = null;
    paint();
  });
  paint();
}

function stopServer() {
  return new Promise((resolve) => {
    if (!child) {
      vscode.window.showInformationMessage(
        "Graph Topology: this editor did not start the server; stop it where it runs.");
      resolve();
      return;
    }
    const pid = child.pid;
    child.once("exit", () => resolve());
    note(`stopping the server (pid ${pid})`);
    if (process.platform === "win32") {
      // the shell and everything under it: python, node
      childProcess.exec(`taskkill /PID ${pid} /T /F`);
    } else {
      try { process.kill(-pid, "SIGTERM"); } catch (_) { child.kill("SIGTERM"); }
    }
    setTimeout(resolve, 8000);
  });
}

async function restartServer() {
  if (child) await stopServer();
  startServer();
}

// -- the conversation's panel ---------------------------------------------------------
function openPanel() {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside, true);
    return;
  }
  panel = vscode.window.createWebviewPanel("graphTopology", "Graph Topology",
    { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    { enableScripts: true, retainContextWhenHidden: true });
  const nonce = crypto.randomBytes(16).toString("hex");
  panel.webview.html = lib.webviewHtml(serverUrl(), sid(), nonce);
  panel.webview.onDidReceiveMessage(fromPage);
  panel.onDidDispose(() => { panel = null; });
  if (!health) panel.webview.postMessage({ type: "down" });
}

function refreshPanel() {
  if (panel) panel.webview.postMessage({ type: "reload" });
}

// What the page asks of the editor: open a file where a function is, or
// put code in -- where the user chooses.
async function fromPage(message) {
  if (message.type === "open" && folder()) {
    const uri = vscode.Uri.joinPath(folder().uri, message.file);
    const document = await vscode.workspace.openTextDocument(uri);
    const line = Math.max(0, (message.line || 1) - 1);
    await vscode.window.showTextDocument(document, {
      viewColumn: vscode.ViewColumn.One,
      selection: new vscode.Range(line, 0, line, 0) });
  } else if (message.type === "insert") {
    await offer(message.code);
  }
}

async function offer(code, selection) {
  const editor = vscode.window.activeTextEditor || lastEditor;
  const picks = [];
  if (editor) {
    picks.push({ label: "$(insert) Insert at the cursor", id: "insert" });
    if (!editor.selection.isEmpty || selection)
      picks.push({ label: "$(replace) Replace the selection", id: "replace" });
    picks.push({ label: "$(arrow-down) Insert below the selection", id: "below" });
  }
  picks.push({ label: "$(new-file) Open beside, as a new file", id: "open" });
  picks.push({ label: "$(copy) Copy", id: "copy" });
  const pick = await vscode.window.showQuickPick(picks, {
    placeHolder: "Where should the code go? Nothing is written until you choose." });
  if (!pick) return;
  if (pick.id === "copy") {
    await vscode.env.clipboard.writeText(code);
  } else if (pick.id === "open") {
    const document = await vscode.workspace.openTextDocument({ content: code, language: "typescript" });
    await vscode.window.showTextDocument(document, vscode.ViewColumn.Beside);
  } else {
    const range = selection || editor.selection;
    await editor.edit((edit) => {
      if (pick.id === "replace") edit.replace(range, code);
      else if (pick.id === "below") edit.insert(new vscode.Position(range.end.line + 1, 0), code + "\n");
      else edit.insert(editor.selection.active, code);
    });
  }
}

// -- the project ----------------------------------------------------------------------
async function readProject(quiet) {
  if (!folder()) throw new Error("open a folder first");
  if (!can("project")) throw new Error(`the server at ${serverUrl()} does not read projects`);
  const most = config().get("maxFiles");
  const largest = config().get("maxFileBytes");
  const uris = await vscode.workspace.findFiles(
    new vscode.RelativePattern(folder(), config().get("projectInclude")),
    config().get("projectExclude"), most + 1);
  const files = {};
  let skipped = 0;
  for (const uri of uris.slice(0, most)) {
    const stat = await vscode.workspace.fs.stat(uri);
    if (stat.size > largest) { skipped += 1; continue; }
    const bytes = await vscode.workspace.fs.readFile(uri);
    files[lib.relative(folder().uri.fsPath, uri.fsPath)] = Buffer.from(bytes).toString("utf8");
  }
  const found = await request("/api/project", { sid: sid(), files, name: folder().name },
                              5 * 60 * 1000);
  projectRead = true;
  paint();
  refreshPanel();
  const said = `read ${found.files} files, ${found.functions} functions, ${found.lines} lines` +
    (skipped ? ` (${skipped} left out: larger than ${largest} bytes)` : "") +
    (uris.length > most ? ` (only the first ${most} files)` : "");
  note(said);
  if (!quiet) vscode.window.showInformationMessage(`Graph Topology: ${said}.`);
}

async function sendFile(uri, text) {
  if (!projectRead || !health || !config().get("syncOnSave") || !folder()) return;
  const path = lib.relative(folder().uri.fsPath, uri.fsPath);
  if (!lib.isRead(path)) return;
  try {
    await request("/api/file", { sid: sid(), path, text });
  } catch (bad) {
    note(`${path} not sent: ${bad.message}`);
  }
}

// -- commands ---------------------------------------------------------------------------
async function ask() {
  const text = await vscode.window.showInputBox({
    prompt: "Ask Graph Topology (about the project, the code, or anything)",
    placeHolder: "what is in the project · who calls main · write a function that …" });
  if (!text) return;
  openPanel();
  const turn = await say(text);
  const reply = (turn.reply || {}).text || (turn.answer || {}).text || "";
  vscode.window.showInformationMessage(reply.slice(0, 400));
}

async function explainSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) return;
  openPanel();
  const turn = await say(lib.asQuestion(editor.document.getText(editor.selection),
                                        editor.document.languageId));
  const reply = (turn.reply || {}).text || (turn.answer || {}).text || "";
  vscode.window.showInformationMessage(reply.slice(0, 500));
}

async function writeFunction() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) return;
  const selection = editor.selection;
  lastEditor = editor;
  openPanel();
  const turn = await say(lib.asRequest(editor.document.getText(selection)));
  const found = ((turn.answer || {}).code || {}).answer || {};
  const reply = (turn.reply || {}).text || (turn.answer || {}).text || "";
  if (!found.code) {
    vscode.window.showWarningMessage(reply.slice(0, 400));
    return;
  }
  vscode.window.showInformationMessage(reply.slice(0, 300));
  await offer(found.code, selection);
}

async function actions() {
  const picks = [
    { label: "$(comment-discussion) Open the conversation", run: openPanel },
    { label: "$(folder-library) Read the project", run: () => readProject(false) },
    { label: "$(question) Ask…", run: ask },
    { label: "$(play) Start the server", run: startServer },
    { label: "$(debug-restart) Restart the server", run: restartServer },
    { label: "$(debug-stop) Stop the server", run: stopServer },
    { label: "$(output) The server's output", run: () => output.show() },
    { label: "$(gear) Settings", run: () => vscode.commands.executeCommand(
      "workbench.action.openSettings", "graphTopology") },
  ];
  const pick = await vscode.window.showQuickPick(picks, {
    placeHolder: health ? `${health.server} at ${serverUrl()}` : `no server at ${serverUrl()}` });
  if (pick) await pick.run();
}

function guarded(run) {
  return async (...args) => {
    try {
      await run(...args);
    } catch (bad) {
      vscode.window.showErrorMessage(`Graph Topology: ${bad.message || bad}`);
    }
  };
}

function activate(context) {
  output = vscode.window.createOutputChannel("Graph Topology");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  status.command = "graphTopology.actions";
  status.show();
  paint();
  const commands = {
    "graphTopology.open": openPanel,
    "graphTopology.readProject": () => readProject(false),
    "graphTopology.ask": ask,
    "graphTopology.explainSelection": explainSelection,
    "graphTopology.writeFunction": writeFunction,
    "graphTopology.startServer": startServer,
    "graphTopology.stopServer": stopServer,
    "graphTopology.restartServer": restartServer,
    "graphTopology.serverOutput": () => output.show(),
    "graphTopology.actions": actions,
  };
  for (const [name, run] of Object.entries(commands))
    context.subscriptions.push(vscode.commands.registerCommand(name, guarded(run)));
  context.subscriptions.push(
    output, status,
    vscode.window.onDidChangeActiveTextEditor((editor) => { if (editor) lastEditor = editor; }),
    vscode.workspace.onDidSaveTextDocument((document) => sendFile(document.uri, document.getText())),
    vscode.workspace.onDidDeleteFiles((event) => event.files.forEach((uri) => sendFile(uri, null))),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("graphTopology.serverUrl")) {
        health = null;
        projectRead = false;
        if (panel) {
          panel.webview.html = lib.webviewHtml(serverUrl(), sid(), crypto.randomBytes(16).toString("hex"));
        }
        check();
      }
    }));
  lastEditor = vscode.window.activeTextEditor;
  timer = setInterval(check, 3000);
  check().then(() => {
    if (!health && config().get("autoStart") && config().get("serverCommand")) startServer();
  });
}

async function deactivate() {
  if (timer) clearInterval(timer);
  if (child) await stopServer();
}

module.exports = { activate, deactivate };
