// What the extension does without the editor, checked under Node:
//     node tools/vscode-graph-topology/test.js
"use strict";

const assert = require("assert");
const lib = require("./lib");

assert.strictEqual(lib.sidFor("C:\\work\\a"), lib.sidFor("C:\\work\\a"));
assert.notStrictEqual(lib.sidFor("C:\\work\\a"), lib.sidFor("C:\\work\\b"));
assert.match(lib.sidFor("x"), /^vscode-[0-9a-f]{12}$/);

assert.strictEqual(lib.asRequest(
  "// sum a list of numbers\n// total([1, 2]) == 3\nfunction total(xs: number[]): number"),
  "sum a list of numbers\ntotal([1, 2]) == 3\nfunction total(xs: number[]): number");
assert.strictEqual(lib.asRequest("/**\n * reverse it\n * rev(\"ab\") == \"ba\"\n */"),
  "reverse it\nrev(\"ab\") == \"ba\"");

assert.strictEqual(lib.asQuestion("function f() {}\n\n", "typescript"),
  "explain this code\n```ts\nfunction f() {}\n```");

assert.strictEqual(lib.base("http://127.0.0.1:8697/"), "http://127.0.0.1:8697");
assert.strictEqual(lib.base(""), "http://127.0.0.1:8697");

assert.deepStrictEqual(lib.compare(null, null), { up: false, restarted: false });
assert.deepStrictEqual(lib.compare(null, { instance: "a" }), { up: true, restarted: false });
assert.deepStrictEqual(lib.compare({ instance: "a" }, { instance: "a" }), { up: true, restarted: false });
assert.deepStrictEqual(lib.compare({ instance: "a" }, { instance: "b" }), { up: true, restarted: true });

assert.strictEqual(lib.relative("C:\\work\\proj", "C:\\work\\proj\\src\\a.ts"), "src/a.ts");
assert.strictEqual(lib.relative("c:/work/proj/", "C:/work/proj/b.js"), "b.js");

assert.ok(lib.isRead("src/a.ts"));
assert.ok(!lib.isRead("README.md"));
assert.ok(!lib.isRead("node_modules/x/index.js"));
assert.ok(!lib.isRead("packages/app/dist/bundle.js"));

const html = lib.webviewHtml("http://127.0.0.1:8697", "vscode-abc", "n0nce");
assert.ok(html.includes('src="http://127.0.0.1:8697/?sid=vscode-abc"'));
assert.ok(html.includes("frame-src http://127.0.0.1:8697"));
assert.ok(html.includes("nonce-n0nce"));

console.log("ok: lib");
