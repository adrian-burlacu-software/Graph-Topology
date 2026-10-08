// What the extension does that does not need the editor: tested under Node
// alone (`test.js`).
"use strict";

const crypto = require("crypto");

/** The conversation this workspace is: the same every time it is opened. */
function sidFor(folder) {
  const hash = crypto.createHash("sha1").update(String(folder || "no folder"))
    .digest("hex").slice(0, 12);
  return `vscode-${hash}`;
}

/** A selection as a request: comment markers taken off, so what was
 *  written as a comment above a function reads as English. */
function asRequest(text) {
  return String(text)
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*(\/\/+|\/\*+|\*\/|\*(?!\/)|#)\s?/, "")
      .replace(/\s*\*\/\s*$/, ""))
    .filter((line, at, all) => line.trim() || (at > 0 && at < all.length - 1))
    .join("\n")
    .trim();
}

/** Code pasted with a question about it: what the server reads as such. */
function asQuestion(code, language) {
  const fence = /python/i.test(language || "") ? "python"
    : /typescript|javascript|ts|js/i.test(language || "") ? "ts" : "";
  return `explain this code\n\`\`\`${fence}\n${String(code).replace(/\s+$/, "")}\n\`\`\``;
}

/** The server's base, with no trailing slash. */
function base(url) {
  return String(url || "http://127.0.0.1:8697").replace(/\/+$/, "");
}

/** What a health reply says, against the last one: whether it is up, and
 *  whether it is the same start of it (`instance`) as before. */
function compare(previous, health) {
  if (!health || health.error) return { up: false, restarted: false };
  return {
    up: true,
    restarted: !!(previous && previous.instance && health.instance
                  && previous.instance !== health.instance),
  };
}

/** A path as the server holds it: relative to the workspace, forward
 *  slashes. */
function relative(folderPath, filePath) {
  const folder = String(folderPath).replace(/\\/g, "/").replace(/\/+$/, "");
  const file = String(filePath).replace(/\\/g, "/");
  return file.toLowerCase().startsWith(folder.toLowerCase() + "/")
    ? file.slice(folder.length + 1) : file;
}

// code, and data (v701): read as what it holds and its schema
const READ = /\.(ts|tsx|js|jsx|mjs|cjs|mts|cts|py|json|ya?ml|csv|tsv)$/i;

/** Whether a file is one the project is read from (for a save). */
function isRead(path, excluded) {
  if (!READ.test(path)) return false;
  return !(excluded || ["node_modules/", "/dist/", "/build/", "/out/", ".git/",
                        "__pycache__/", ".venv/", "/venv/", "site-packages/"])
    .some((part) => ("/" + path.replace(/\\/g, "/")).includes(part.startsWith("/") ? part : "/" + part));
}

/** The webview's page: the server's own page in a frame, the frame's
 *  messages passed on to the editor. */
function webviewHtml(url, sid, nonce) {
  const origin = new URL(base(url)).origin;
  const page = `${base(url)}/?sid=${encodeURIComponent(sid)}`;
  return `<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src ${origin} http://127.0.0.1:* http://localhost:*; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>html,body,iframe{margin:0;padding:0;border:0;width:100%;height:100%;overflow:hidden;background:#0e1116}
#down{display:none;color:#d9e0e8;font:14px system-ui;padding:24px}</style></head>
<body><iframe id="page" src="${page}" allow="clipboard-read; clipboard-write"></iframe>
<div id="down">No server at ${origin}. Start one (Graph Topology: Start the Server) or set graphTopology.serverUrl.</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const frame = document.getElementById("page");
window.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.source === "graph-topology") { vscode.postMessage(data); return; }
  if (data.type === "reload") { frame.src = "${page}&t=" + Date.now(); }
  if (data.type === "down") { frame.style.display = "none"; document.getElementById("down").style.display = "block"; }
  if (data.type === "up") { frame.style.display = "block"; document.getElementById("down").style.display = "none"; }
});
</script></body></html>`;
}

/** The editor's language for a piece of code: Python's `def`, else
    TypeScript (what a new document beside is opened as). */
function languageOf(code) {
  return /^\s*(async\s+)?def\s+\w+\s*\(/m.test(String(code))
    && !/\bfunction\s+\w+\s*\(/.test(String(code)) ? "python" : "typescript";
}

module.exports = { sidFor, asRequest, asQuestion, languageOf, base, compare, relative,
                   isRead, webviewHtml };
