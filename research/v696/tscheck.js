// The TypeScript checker: one persistent process, JSON lines in and out.
//
//   {"id": 1, "op": "check", "source": "..."}
//       -> {"id": 1, "ok": true|false, "errors": ["..."]}
//   {"id": 2, "op": "run", "source": "...", "entry": "f",
//    "cases": [[args...], ...], "timeout": 200}
//       -> {"id": 2, "ok": true, "outputs": [value | {"error": "..."}]}
//   {"id": 3, "op": "tests", "source": "...", "timeout": 2000}
//       -> {"id": 3, "ok": true|false, "error": "..."}   (source runs whole)
//
// Types are checked by the compiler API over an in-memory program whose
// library files are read once and kept, so a check costs what the
// candidate costs. Code runs in a fresh `vm` context with a timeout.
"use strict";

const path = require("path");
const fs = require("fs");
const vm = require("vm");
const readline = require("readline");
const ts = require("typescript");

const LIB_DIR = path.dirname(require.resolve("typescript/lib/lib.d.ts"));
const OPTIONS = {
  target: ts.ScriptTarget.ES2020,
  lib: ["lib.es2020.d.ts"],
  strict: true,
  noEmit: true,
  types: [],
};
// What is read (`tree`) may say more than the search grows: a program a
// person wrote uses what the language has since had (`xs.at(-1)`). The
// library the search grows from (`signatures`) stays what it was.
const READING = Object.assign({}, OPTIONS, { lib: ["lib.es2022.d.ts"] });
const LIBS = new Map();

function libFile(name) {
  if (!LIBS.has(name)) {
    const file = path.join(LIB_DIR, name);
    const text = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : undefined;
    LIBS.set(name, text === undefined ? undefined
      : ts.createSourceFile(name, text, ts.ScriptTarget.ES2020, true));
  }
  return LIBS.get(name);
}

// `declare var require: any;` is what MultiPL-E's tests start with; the
// candidate is checked on its own, with that declared so it may be used.
const PRELUDE = "declare var require: any;\n";

function check(source) {
  const name = "candidate.ts";
  const text = PRELUDE + source;
  const host = {
    getSourceFile: (file) => file === name
      ? ts.createSourceFile(name, text, ts.ScriptTarget.ES2020, true)
      : libFile(path.basename(file)),
    getDefaultLibFileName: () => "lib.es2020.d.ts",
    writeFile: () => {},
    getCurrentDirectory: () => "/",
    getDirectories: () => [],
    fileExists: (file) => file === name || libFile(path.basename(file)) !== undefined,
    readFile: (file) => file === name ? text : undefined,
    getCanonicalFileName: (file) => file,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
  };
  const program = ts.createProgram([name], OPTIONS, host);
  const diagnostics = ts.getPreEmitDiagnostics(program)
    .filter((one) => one.file && one.file.fileName === name);
  return diagnostics.map((one) =>
    ts.flattenDiagnosticMessageText(one.messageText, "\n"));
}

function transpile(source) {
  return ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2020,
                       module: ts.ModuleKind.CommonJS },
  }).outputText;
}

// A value as it is reported: what JSON would lose is kept -- a Set's or a
// Map's contents, an infinite number -- and what is not a value to compare
// (a function) is said to be so, not dropped.
function plain(value) {
  if (value === undefined) return null;
  if (typeof value === "number" && !isFinite(value)) return String(value);
  if (typeof value === "function" || typeof value === "symbol")
    throw new Error("not a value");
  // by its tag: a value made in the sandbox has the sandbox's own Set
  const tag = Object.prototype.toString.call(value);
  if (tag === "[object Set]") return { $set: [...value].map(plain) };
  if (tag === "[object Map]")
    return { $map: [...value].map(([k, v]) => [plain(k), plain(v)]) };
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value)) out[key] = plain(value[key]);
    return out;
  }
  return value;
}

function reported(value) {
  try {
    const out = plain(value);
    // a value past this is not one to compare: a runaway, said so
    if (JSON.stringify(out).length > LARGEST_VALUE)
      return { error: "a value too large" };
    return { value: out };
  } catch (error) {
    return { error: String(error && error.message || error) };
  }
}

function sandbox() {
  // what a program prints is kept, a line a call (`run` reports it):
  // printing is something a program does
  const context = { require, module: {}, exports: {}, __printed: [] };
  const say = (...parts) => {
    if (context.__printed.length < 200)
      context.__printed.push(parts.map((one) => typeof one === "string"
        ? one : JSON.stringify(one)).join(" "));
  };
  context.console = { log: say, info: say, warn: say, error: say };
  return vm.createContext(context);
}

function run(source, entry, cases, timeout) {
  const context = sandbox();
  vm.runInContext(transpile(source) + `\n;globalThis.__entry = ${entry};`,
                  context, { timeout });
  // One case run out of time is the program failing: the rest are not run
  // (a loop that never ends would cost every case its timeout, for every
  // edit of a program that has one).
  let out = null;
  return cases.map((args) => {
    if (out) return { error: out };
    context.__printed = [];
    const printed = (reply) => context.__printed.length
      ? { ...reply, printed: context.__printed.slice() } : reply;
    try {
      context.__args = args;
      const value = vm.runInContext("__entry(...__args)", context,
                                    { timeout });
      return printed(reported(value));
    } catch (error) {
      const said = String(error && error.message || error);
      if (/timed out/.test(said)) out = said;
      return printed({ error: said });
    }
  });
}

// The library as the compiler has it: for each receiver type named, every
// method's parameters and return type, with the receiver's type variables
// instantiated (`number[]`'s `indexOf(searchElement: number, ...)`), and
// the global functions asked for. Read from TypeScript's own lib files --
// nothing about what a method does is written here.
function signatures(receivers, globals) {
  const name = "probe.ts";
  const lines = receivers.map((type, index) => `declare const r${index}: ${type};`);
  const text = lines.join("\n") + "\n";
  const host = {
    getSourceFile: (file) => file === name
      ? ts.createSourceFile(name, text, ts.ScriptTarget.ES2020, true)
      : libFile(path.basename(file)),
    getDefaultLibFileName: () => "lib.es2020.d.ts",
    writeFile: () => {},
    getCurrentDirectory: () => "/",
    getDirectories: () => [],
    fileExists: (file) => file === name || libFile(path.basename(file)) !== undefined,
    readFile: (file) => file === name ? text : undefined,
    getCanonicalFileName: (file) => file,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
  };
  const program = ts.createProgram([name], OPTIONS, host);
  const checker = program.getTypeChecker();
  const source = program.getSourceFile(name);
  const out = [];
  const describe = (signature, node) => ({
    params: signature.getParameters().map((symbol) => {
      const declaration = symbol.valueDeclaration;
      return {
        name: symbol.getName(),
        type: checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, node)),
        optional: !!(declaration && (declaration.questionToken || declaration.initializer)),
        rest: !!(declaration && declaration.dotDotDotToken),
      };
    }),
    returns: checker.typeToString(signature.getReturnType()),
    generic: !!(signature.typeParameters && signature.typeParameters.length),
  });
  source.statements.forEach((statement, index) => {
    const declaration = statement.declarationList.declarations[0];
    const type = checker.getTypeAtLocation(declaration.name);
    for (const property of checker.getPropertiesOfType(type)) {
      const member = checker.getTypeOfSymbolAtLocation(property, declaration.name);
      const calls = member.getCallSignatures();
      const deprecated = property.getJsDocTags()
        .some((tag) => tag.name === "deprecated");
      const doc = ts.displayPartsToString(
        property.getDocumentationComment(checker));
      if (calls.length === 0) {
        out.push({ receiver: receivers[index], name: property.getName(),
                   property: true, params: [], deprecated, doc,
                   returns: checker.typeToString(member), generic: false });
        continue;
      }
      calls.forEach((signature) => out.push(Object.assign(
        { receiver: receivers[index], name: property.getName(),
          property: false, deprecated, doc }, describe(signature, declaration.name))));
    }
  });
  const scope = checker.getSymbolsInScope(source, ts.SymbolFlags.Value);
  for (const symbol of scope) {
    if (!globals.includes(symbol.getName())) continue;
    const type = checker.getTypeOfSymbolAtLocation(symbol, source);
    for (const property of [null].concat(symbol.getName() === "Math"
        ? checker.getPropertiesOfType(type) : [])) {
      const owner = property === null ? type
        : checker.getTypeOfSymbolAtLocation(property, source);
      owner.getCallSignatures().forEach((signature) => out.push(Object.assign(
        { receiver: null, deprecated: property !== null && property.getJsDocTags()
            .some((tag) => tag.name === "deprecated"),
          name: property === null ? symbol.getName()
            : `Math.${property.getName()}`, property: false,
          doc: ts.displayPartsToString((property || symbol)
            .getDocumentationComment(checker)) },
        describe(signature, source))));
    }
  }
  return out;
}

// Many expressions at once over the same parameters and cases: each
// expression's value on each case, or its error. The candidates of a
// search are evaluated here in batches, so what code does is always what
// Node says it does, and never a copy of it.
function values(params, cases, expressions, timeout, prelude) {
  const context = sandbox();
  context.__cases = cases;
  // The helpers the expressions call, declared first (rung 3). Helpers that
  // do not compile or run are every expression failing, not the checker.
  if (prelude) {
    try {
      vm.runInContext(transpile(prelude), context, { timeout });
    } catch (error) {
      const said = "helpers: " + String(error && error.message || error);
      return expressions.map(() => cases.map(() => ({ error: said })));
    }
  }
  const out = [];
  for (const expression of expressions) {
    let compiled;
    try {
      compiled = vm.runInContext(
        `(function(${params.join(", ")}) { return (${expression}); })`,
        context, { timeout });
    } catch (error) {
      out.push(cases.map(() => ({ error: "compile" })));
      continue;
    }
    context.__f = compiled;
    const row = [];
    for (let index = 0; index < cases.length; index++) {
      try {
        context.__i = index;
        const value = vm.runInContext(
          "__f(...JSON.parse(JSON.stringify(__cases[__i])))", context,
          { timeout });
        row.push(reported(value));
      } catch (error) {
        const said = String(error && error.message || error).slice(0, 80);
        row.push({ error: said });
        if (/timed out/.test(said)) {
          // One case run out of time is the candidate failing: the rest
          // are not run (a search's loop that never ends would otherwise
          // cost every case its timeout, batch after batch).
          while (row.length < cases.length) row.push({ error: said });
          break;
        }
      }
    }
    out.push(row);
  }
  return out;
}

function tests(source, timeout) {
  vm.runInContext(transpile(source), sandbox(), { timeout });
}

// A project: several files, one program, imports resolved among them.
function programOfFiles(files, options) {
  const texts = new Map(Object.entries(files).map(([file, text]) =>
    [file, PRELUDE + text]));
  const host = {
    getSourceFile: (file) => texts.has(file)
      ? ts.createSourceFile(file, texts.get(file), ts.ScriptTarget.ES2020, true)
      : libFile(path.basename(file)),
    getDefaultLibFileName: () => "lib.es2020.d.ts",
    writeFile: () => {},
    getCurrentDirectory: () => "/",
    getDirectories: () => [],
    fileExists: (file) => texts.has(file)
      || libFile(path.basename(file)) !== undefined,
    readFile: (file) => texts.get(file),
    getCanonicalFileName: (file) => file,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
  };
  return ts.createProgram([...texts.keys()],
    Object.assign({}, options || OPTIONS, { moduleResolution:
                                 ts.ModuleResolutionKind.NodeJs }), host);
}

// What the compiler says is wrong in a project, and where: each error's
// file, span (in the file as given) and message.
function diagnose(files) {
  const program = programOfFiles(files);
  const at = PRELUDE.length;
  return ts.getPreEmitDiagnostics(program)
    .filter((one) => one.file && one.file.fileName in files)
    .map((one) => ({ file: one.file.fileName, code: one.code,
                     start: one.start - at, end: one.start + one.length - at,
                     message: ts.flattenDiagnosticMessageText(one.messageText,
                                                              "\n") }));
}

// A project run: each file a CommonJS module, `require` finding the
// project's own files; the file `main` run whole (its tests). The loader
// runs inside the sandbox, as one script under its timeout: a module that
// never ends (a helper made wrong) is stopped like any candidate, not run
// by the checker itself where no timeout reaches.
const LOADER = `
(function (files, main) {
  const cache = {};
  const resolve = (from, spec) => {
    const parts = from.split("/").slice(0, -1);
    for (const part of spec.split("/")) {
      if (part === "..") parts.pop();
      else if (part !== ".") parts.push(part);
    }
    const base = parts.join("/");
    if (base in files) return base;
    if (base + ".ts" in files) return base + ".ts";
    throw new Error("no module " + spec + " from " + from);
  };
  const load = (file) => {
    if (cache[file]) return cache[file].exports;
    const module = { exports: {} };
    cache[file] = module;
    const wrapped = new Function("require", "module", "exports", files[file]);
    wrapped((spec) => spec.startsWith(".") ? load(resolve(file, spec))
                                          : require(spec),
            module, module.exports);
    return module.exports;
  };
  load(main);
})(__files, __main);
`;

function project(files, main, timeout) {
  const context = sandbox();
  const transpiled = {};
  for (const [file, text] of Object.entries(files))
    transpiled[file] = transpile(text);
  context.__files = transpiled;
  context.__main = main;
  vm.runInContext(LOADER, context, { timeout });
}

function programOf(name, text, options) {
  const host = {
    getSourceFile: (file) => file === name
      ? ts.createSourceFile(name, text, ts.ScriptTarget.ES2020, true)
      : libFile(path.basename(file)),
    getDefaultLibFileName: () => "lib.es2020.d.ts",
    writeFile: () => {},
    getCurrentDirectory: () => "/",
    getDirectories: () => [],
    fileExists: (file) => file === name || libFile(path.basename(file)) !== undefined,
    readFile: (file) => file === name ? text : undefined,
    getCanonicalFileName: (file) => file,
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
  };
  return ts.createProgram([name], options || OPTIONS, host);
}

// JavaScript read as TypeScript: what it leaves unsaid is said for it, from
// how it is used -- a helper's parameter takes the type of what it is first
// called with, a name assigned and never declared is declared where its
// function begins. Only text is added, and `origin` says where each
// character of the completed text was in the text as given, so every span
// read is a span of the source as it was written.
function completed(name, text) {
  const origin = Array.from({ length: text.length + 1 }, (_, at) => at);
  for (let pass = 0; pass < 3; pass++) {
    const program = programOf(name, text, READING);
    const checker = program.getTypeChecker();
    const file = program.getSourceFile(name);
    const vague = (node) => /\bany\b/.test(checker.typeToString(
      checker.getBaseTypeOfLiteralType(checker.getTypeAtLocation(node))));
    const inserts = new Map();
    // a helper's parameters: typed by the first call that says
    const owners = new Map();
    (function walk(node) {
      if ((ts.isFunctionDeclaration(node) && node.name && node.body)
          || (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)
              && node.initializer && (ts.isArrowFunction(node.initializer)
                || ts.isFunctionExpression(node.initializer)))) {
        const symbol = checker.getSymbolAtLocation(node.name);
        const made = ts.isFunctionDeclaration(node) ? node : node.initializer;
        if (symbol && made.parameters.some((one) => !one.type
            && ts.isIdentifier(one.name) && !one.initializer
            && !one.dotDotDotToken))
          owners.set(symbol, made);
      }
      node.forEachChild(walk);
    })(file);
    (function walk(node) {
      if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
        const made = owners.get(checker.getSymbolAtLocation(node.expression));
        if (made) made.parameters.forEach((one, at) => {
          const given = node.arguments[at];
          if (one.type || !ts.isIdentifier(one.name) || one.initializer
              || one.dotDotDotToken || !given || ts.isSpreadElement(given)
              || vague(given) || inserts.has(one.name.getEnd()))
            return;
          const type = checker.typeToString(checker.getBaseTypeOfLiteralType(
            checker.getTypeAtLocation(given)));
          if (/=>|\{|undefined|null|never/.test(type)) return;
          // `x => ...` is `(x: T) => ...` once it says a type
          const bare = ts.isArrowFunction(made)
            && made.getStart() === one.getStart();
          inserts.set(one.name.getEnd(), bare ? `: ${type})` : `: ${type}`);
          if (bare) inserts.set(one.name.getStart(), "(");
        });
      }
      node.forEachChild(walk);
    })(file);
    // a list begun empty that the compiler never settles (its elements
    // set from its own): it holds what is first put into it, or -- being
    // what the function returns -- what the function says it returns
    (function walk(node) {
      if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)
          && !node.type && (!node.initializer
            || (ts.isArrayLiteralExpression(node.initializer)
                && node.initializer.elements.length === 0))) {
        const symbol = checker.getSymbolAtLocation(node.name);
        let scope = node;
        while (scope.parent && !ts.isFunctionLike(scope)) scope = scope.parent;
        let unsettled = false, empty = !!node.initializer;
        let put = null, given = null;
        (function inside(one) {
          if (ts.isIdentifier(one) && one !== node.name
              && checker.getSymbolAtLocation(one) === symbol) {
            const parent = one.parent;
            if (vague(one)) unsettled = true;
            if (ts.isBinaryExpression(parent) && parent.left === one
                && parent.operatorToken.kind === ts.SyntaxKind.EqualsToken
                && ts.isArrayLiteralExpression(parent.right)
                && parent.right.elements.length === 0) empty = true;
            if (!put && ts.isPropertyAccessExpression(parent)
                && parent.expression === one && parent.name.text === "push"
                && ts.isCallExpression(parent.parent)
                && parent.parent.arguments.length === 1
                && !vague(parent.parent.arguments[0])) {
              const type = checker.typeToString(
                checker.getBaseTypeOfLiteralType(checker.getTypeAtLocation(
                  parent.parent.arguments[0])));
              if (!/=>|\{|undefined|null|never/.test(type))
                put = /[ |]/.test(type) ? `(${type})[]` : `${type}[]`;
            }
            if (!given && ts.isReturnStatement(parent)
                && ts.isFunctionLike(scope) && scope.type
                && ts.findAncestor(parent, ts.isFunctionLike) === scope
                && /\[\]$/.test(scope.type.getText()))
              given = scope.type.getText();
          }
          one.forEachChild(inside);
        })(scope);
        if (unsettled && empty && (put || given))
          inserts.set(node.name.getEnd(), `: ${put || given}`);
      }
      node.forEachChild(walk);
    })(file);
    // a name assigned and never declared: declared where its function
    // begins (what it is in JavaScript: a name, from then on)
    const missing = new Map();
    for (const one of ts.getPreEmitDiagnostics(program)) {
      if (one.code !== 2304 || !one.file || one.file.fileName !== name)
        continue;
      let node = ts.getTokenAtPosition(file, one.start);
      if (!ts.isIdentifier(node) || !ts.isBinaryExpression(node.parent)
          || node.parent.left !== node || node.parent.operatorToken.kind
          !== ts.SyntaxKind.EqualsToken
          || !ts.isExpressionStatement(node.parent.parent))
        continue;
      let outer = null;
      for (let up = node.parent; up; up = up.parent)
        if (ts.isFunctionLike(up) && up.body && ts.isBlock(up.body))
          outer = up;
      if (!outer) continue;
      const at = outer.body.getStart() + 1;
      if (!missing.has(at)) missing.set(at, new Set());
      missing.get(at).add(node.text);
    }
    for (const [at, names] of missing)
      inserts.set(at, ` let ${[...names].join(", ")};`);
    // nothing more to say: this is the program that is read
    if (!inserts.size) return { text, origin, program };
    for (const at of [...inserts.keys()].sort((a, b) => b - a)) {
      const said = inserts.get(at);
      text = text.slice(0, at) + said + text.slice(at);
      origin.splice(at, 0, ...Array(said.length).fill(origin[at]));
    }
  }
  return { text, origin, program: programOf(name, text, READING) };
}

// What a function is made of, as the compiler resolves it: every member
// called or read, named by the interface that declares it (`String.split`,
// `Array.filter`, `Math.max` -- whatever the receiver was called), the
// operators, and the statements. And what gives its result: the outermost
// thing its last `return` returns. Nothing here knows what a member does.
class Unread extends Error {}

// The largest tree node a reading copies, and the largest value reported,
// in characters of JSON.
const LARGEST = 5000000;
const LARGEST_VALUE = 100000;

// The methods that change what they are called on.
const CHANGING = ["push", "pop", "shift", "unshift", "splice", "sort",
                  "reverse", "fill", "copyWithin", "add", "delete", "set",
                  "clear"];

const READONLY = { ReadonlyArray: "Array", ReadonlySet: "Set",
                   ReadonlyMap: "Map" };
const OPERATOR = { "==": "===", "!=": "!==" };
const STATEMENTS = {
  ForStatement: "for", ForOfStatement: "for of", ForInStatement: "for in",
  WhileStatement: "while", DoStatement: "while", IfStatement: "if",
  SwitchStatement: "switch", TryStatement: "try",
  ConditionalExpression: "?:", ArrowFunction: "=>",
  FunctionExpression: "=>", ArrayLiteralExpression: "[]",
  ObjectLiteralExpression: "{}", SpreadElement: "...",
  TemplateExpression: "template", ElementAccessExpression: "[i]",
  VariableDeclaration: "let", RegularExpressionLiteral: "regex",
  BreakStatement: "break", ContinueStatement: "continue",
};

// A function whose body is one `return` as a typed tree, every node with
// the type the compiler gives it (literal types widened): what is read back
// into the search's own trees (`program.parse`). Anything else is
// {"k": "other"} and the reading fails there.
function tree(source, files) {
  // one file, its functions by name; or a project, its functions by
  // "file#name", a call through an import resolved to where it is declared
  const name = "tree.ts";
  const inProject = !!files;
  const whole = inProject ? null : completed(name, PRELUDE + source);
  const program = inProject ? programOfFiles(files, READING)
    : whole.program;
  const checker = program.getTypeChecker();
  const ours = new Set(inProject ? Object.keys(files) : [name]);
  const isOurs = (sourceFile) => ours.has(sourceFile.fileName);
  const keyOf = (sourceFile, fname) => inProject
    ? `${sourceFile.fileName}#${fname}` : fname;
  let current = inProject ? null : name;
  const typeOf = (node) => checker.typeToString(
    checker.getBaseTypeOfLiteralType(checker.getTypeAtLocation(node)));
  const memberName = (node) => {
    const symbol = checker.getSymbolAtLocation(node.name || node);
    const declaration = symbol && symbol.declarations && symbol.declarations[0];
    const owner = declaration && declaration.parent;
    let where = owner && owner.name && owner.name.text;
    if (!where) return null;
    where = READONLY[where] || where;
    return `${where.replace(/Constructor$/, "")}.${symbol.getName()}`;
  };
  // The body is read by executing it symbolically: a local variable's
  // name, read, is its value at that point (`env`), so steps, branches and
  // loops all come out as one expression over the parameters.
  let env = new Map();
  const declared = new Map();
  // A value is copied wherever its name is read: a program whose values
  // are built from values built from values grows as it is read (a guard
  // reading a variable other guards set). Past this, it is not read as one
  // expression -- said so, not read for an hour.
  const copy = (value) => {
    const text = JSON.stringify(value);
    if (text.length > LARGEST)
      throw new Unread("a value too large to copy where it is read");
    return JSON.parse(text);
  };

  // A large value is not copied where its name is read: it is bound once,
  // where it was made -- the function's body, a loop's body, a callback's
  // -- and every read refers to the binding. The binding is lazy and
  // remembers: its value is worked out when first read and not before (a
  // value an `if` guards may not even be computable otherwise), and once.
  // So a program is read in the size it was written.
  // how many nodes make a value large enough to bind rather than copy
  const SHARED = 12;
  let frames = [];
  let shares = 0;
  const enterFrame = () => {
    const frame = { defs: [], inherited: new Set(env.values()) };
    frames.push(frame);
    return frame;
  };
  const leaveFrame = (frame, result) => {
    frames.splice(frames.indexOf(frame), 1);
    let out = result;
    for (const def of [...frame.defs].reverse()) {
      // bound and then never read (a way through that was read and left):
      // not part of what the function is
      if (!JSON.stringify(out).includes(`"key":"${def.key}"`)) continue;
      out = { k: "let", key: def.key, value: def.value, body: out,
              type: out.type };
    }
    return out;
  };
  const fetch = (name) => share(env.get(name));
  // A value put inside another -- read by its name, or kept as the other
  // side of a merged `if` -- is shared the same way: never copied whole
  // more than once.
  const share = (value) => {
    const text = JSON.stringify(value);
    // a loop is work however few words say it
    const costly = /"k":"(while|arrow|range)"/.test(text);
    if (!value || typeof value !== "object")
      return JSON.parse(text);
    // made in the innermost frame that did not inherit it
    let home = null;
    for (let at = frames.length - 1; at >= 0; at--) {
      if (!frames[at].inherited.has(value)) {
        home = frames[at];
        break;
      }
    }
    // Work done before a loop and read inside it is done once, not every
    // time round: bound where it was made, at its first read.
    const inside = costly && home && home !== frames[frames.length - 1];
    if (!inside && (text.match(/"k":/g) || []).length <= SHARED)
      return JSON.parse(text);
    if (!value.$share && !value.$read && !inside) {
      // read once: nothing to share yet -- it is where it is read. A
      // second read binds it (so no value is ever more than twice over).
      Object.defineProperty(value, "$read", { value: true,
                                              enumerable: false });
      return JSON.parse(text);
    }
    if (!value.$share) {
      if (!home) return copy(value);
      const key = `$v${++shares}`;
      home.defs.push({ key, value: JSON.parse(text) });
      Object.defineProperty(value, "$share", { value: key,
                                               enumerable: false });
    }
    return { k: "ref", key: value.$share, type: value.type };
  };
  // Where each node came from (rung 4): its span in the source as given,
  // and for an operator or a member the span of the word itself -- what an
  // edit replaces. A value read from a name keeps the span it was made at.
  const at = PRELUDE.length;
  const placed = (position) => (whole ? whole.origin[position] : position)
    - at;
  const span = (node) => [placed(node.getStart()), placed(node.getEnd())];
  const read = (node) => {
    const out = readNode(node);
    if (out && typeof out === "object" && !out.span) {
      out.span = span(node);
      if (inProject) out.file = current;
      let inner = node;
      while (ts.isParenthesizedExpression(inner) || ts.isAsExpression(inner)
             || ts.isNonNullExpression(inner)) inner = inner.expression;
      if (ts.isBinaryExpression(inner))
        out.opspan = span(inner.operatorToken);
      else if (ts.isPrefixUnaryExpression(inner))
        out.opspan = [placed(inner.getStart()),
                      placed(inner.operand.getStart())];
      else if (ts.isCallExpression(inner)
               && ts.isPropertyAccessExpression(inner.expression))
        out.namespan = span(inner.expression.name);
      else if (ts.isCallExpression(inner) && ts.isIdentifier(inner.expression))
        out.namespan = span(inner.expression);
      else if (ts.isPropertyAccessExpression(inner))
        out.namespan = span(inner.name);
      if (ts.isIdentifier(inner)) out.said = inner.text;
    }
    return out;
  };
  const readNode = (node) => {
    while (ts.isParenthesizedExpression(node) || ts.isAsExpression(node)
           || ts.isNonNullExpression(node)) node = node.expression;
    const type = typeOf(node);
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      const args = node.arguments.map(read);
      if (ts.isPropertyAccessExpression(callee)) {
        const member = memberName(callee);
        if (callee.expression.getText() === "Math")
          return { k: "call", member, recv: null, args, type };
        return { k: "call", member, recv: read(callee.expression), args,
                 type };
      }
      if (ts.isIdentifier(callee)) {
        const bound = env.get(callee.text);
        if (bound && bound.k === "arrow")
          // a helper made inside the function: its body in the call's place
          return { k: "inline", fn: copy(bound), args, type };
        if (bound)
          // a value called that the tree does not model as a function
          // (what `require` gave): the call as it was written
          return opaque(node, type);
        let symbol = checker.getSymbolAtLocation(callee);
        // through an import, to where the function is declared
        if (symbol && (symbol.flags & ts.SymbolFlags.Alias))
          symbol = checker.getAliasedSymbol(symbol);
        const where = symbol && symbol.declarations && symbol.declarations[0];
        // what the language itself declares (`isNaN`, `Number`), as against
        // a helper of this file or project
        // (`require` is declared for every file read, before its text)
        const global = !!(where && (!isOurs(where.getSourceFile())
          || (!inProject && where.getEnd() <= at)));
        const member = where && !global && inProject
          ? keyOf(where.getSourceFile(), symbol.getName()) : callee.text;
        return { k: "call", member, recv: null, args, type, global };
      }
    } else if (ts.isPropertyAccessExpression(node)) {
      return { k: "prop", member: memberName(node),
               recv: read(node.expression), type };
    } else if (ts.isBinaryExpression(node)) {
      const token = ts.tokenToString(node.operatorToken.kind);
      // `==` between two of one plain kind is `===`; between any others
      // it is what it is (`x == null` is true of what is undefined)
      const plain = ["number", "string", "boolean"].includes(
        typeOf(node.left)) && typeOf(node.left) === typeOf(node.right);
      return { k: "bin", op: plain ? OPERATOR[token] || token : token,
               args: [read(node.left), read(node.right)], type };
    } else if (ts.isPrefixUnaryExpression(node)) {
      if (node.operator === ts.SyntaxKind.MinusToken
          && ts.isNumericLiteral(node.operand))
        return { k: "lit", value: -Number(node.operand.text), type };
      return { k: "pre", op: ts.tokenToString(node.operator),
               args: [read(node.operand)], type };
    } else if (ts.isConditionalExpression(node)) {
      return { k: "cond", args: [read(node.condition), read(node.whenTrue),
                                 read(node.whenFalse)], type };
    } else if (ts.isArrowFunction(node) || ts.isFunctionExpression(node)
               || (ts.isFunctionDeclaration(node) && node.body)) {
      const params = node.parameters.map((one) => one.name.getText());
      const outer = env;
      const outerLoops = loops;
      const outerResult = declared.get(RESULT);
      env = new Map(env);
      for (const one of params) env.delete(one);
      loops = [];
      const frame = enterFrame();
      try {
        // what it changes outside itself is not in what it returns
        for (const name of outer.keys())
          if (!name.includes("$") && !params.includes(name)
              && assigned(node.body, name))
            throw new Unread("a callback that changes what is outside it");
        if (!ts.isBlock(node.body))
          return { k: "arrow", params,
                   body: leaveFrame(frame, read(node.body)), type };
        // a function of its own: what it returns is its own
        const signature = checker.getSignatureFromDeclaration(node);
        begin(checker.typeToString(checker.getBaseTypeOfLiteralType(
          signature.getReturnType())));
        const out = run(node.body.statements, 0);
        return { k: "arrow", params,
                 body: leaveFrame(frame, out.ret || fallen()), type };
      } catch (error) {
        if (!(error instanceof Unread)) throw error;
      } finally {
        if (frames.includes(frame)) frames.splice(frames.indexOf(frame), 1);
        env = outer;
        loops = outerLoops;
        declared.set(RESULT, outerResult);
      }
      return opaque(node, type);
    } else if (ts.isElementAccessExpression(node)) {
      return { k: "index", args: [read(node.expression),
                                  read(node.argumentExpression)], type };
    } else if (ts.isIdentifier(node)) {
      if (env.has(canon(node.text))) return fetch(canon(node.text));
      if (node.text === "Infinity")
        return { k: "lit", value: "Infinity", type: "number" };
      const symbol = checker.getSymbolAtLocation(node);
      const where = symbol && symbol.declarations && symbol.declarations[0];
      if (!where || !isOurs(where.getSourceFile())
          // a `let` read before the line that declares it
          || (ts.isVariableDeclaration(where)
              && ts.isVariableStatement(where.parent.parent)
              && (where.parent.flags & ts.NodeFlags.BlockScoped)
              && where.getStart() > node.getStart()
              && ts.findAncestor(where, ts.isFunctionLike)
                 === ts.findAncestor(node, ts.isFunctionLike)))
        // what the language declares, used as a value: `String`, `NaN` --
        // or a name nothing declares here (a program that is wrong says
        // such things): its own text, which fails as it did
        return { k: "opaque", text: node.text, holes: [], type };
      return { k: "id", name: node.text, type };
    } else if (ts.isNumericLiteral(node)) {
      return { k: "lit", value: Number(node.text), type };
    } else if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      return { k: "lit", value: node.text, type };
    } else if (node.kind === ts.SyntaxKind.TrueKeyword
               || node.kind === ts.SyntaxKind.FalseKeyword) {
      return { k: "lit", value: node.kind === ts.SyntaxKind.TrueKeyword, type };
    } else if (ts.isSpreadElement(node)) {
      return { k: "spread", args: [read(node.expression)], type };
    }
    return opaque(node, type);
  };
  // What the tree does not model -- a literal object or list, a regular
  // expression, `new Set(...)`, a template -- is kept as its own text,
  // exactly, with every variable it reads from outside it a hole, filled
  // with what that variable is at this point (so it runs as it did).
  const opaque = (node, type) => {
    const base = node.getStart();
    const holes = [];
    (function walk(one) {
      if (ts.isIdentifier(one)) {
        const parent = one.parent;
        const named = parent && ((ts.isPropertyAccessExpression(parent)
                                  && parent.name === one)
          || (ts.isPropertyAssignment(parent) && parent.name === one));
        const symbol = !named && checker.getSymbolAtLocation(one);
        const where = symbol && symbol.valueDeclaration;
        if (where && (ts.isParameter(where) || ts.isVariableDeclaration(where)
                      || ts.isFunctionDeclaration(where))
            && !(where.getStart() >= node.getStart()
                 && where.getEnd() <= node.getEnd())
            && isOurs(where.getSourceFile())) {
          if (ts.isShorthandPropertyAssignment(parent))
            throw new Unread("a shorthand property");
          holes.push({ at: [one.getStart() - base, one.getEnd() - base],
                       value: read(one) });
          return;
        }
      }
      one.forEachChild(walk);
    })(node);
    return { k: "opaque", text: node.getText(), holes, type };
  };

  // A second name for the same list (`let p = arr`, and one of them then
  // changed in place) is kept in `env` as that: every read and change of
  // it is of the list it names.
  const canon = (text) => {
    let value = env.get(text);
    for (let hops = 0; value && value.k === "alias" && hops < 8; hops++) {
      text = value.to;
      value = env.get(text);
    }
    return text;
  };
  // What a piece of code changes: the name at the root of everything it
  // assigns, counts up or down, deletes from, or calls a changing method
  // on (`dp[i][j] = v` changes `dp`). `unknown` where it may change what
  // has no name here: a call into this file's own functions, a changing
  // method on what a call gave.
  const writes = (body) => {
    const roots = [];
    let unknown = false;
    const root = (target) => {
      while (ts.isElementAccessExpression(target)
             || ts.isPropertyAccessExpression(target)
             || ts.isParenthesizedExpression(target)
             || ts.isNonNullExpression(target)) target = target.expression;
      if (ts.isIdentifier(target)) roots.push(target);
      else if (ts.isArrayLiteralExpression(target)
               || ts.isObjectLiteralExpression(target))
        // unpacked into: every name in the pattern
        (function each(node) {
          if (ts.isIdentifier(node)) roots.push(node);
          node.forEachChild(each);
        })(target);
      else unknown = true;
    };
    (function walk(node) {
      if (ts.isBinaryExpression(node) && node.operatorToken.kind
          >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind
          <= ts.SyntaxKind.LastAssignment) root(node.left);
      else if ((ts.isPrefixUnaryExpression(node)
                || ts.isPostfixUnaryExpression(node))
               && (node.operator === ts.SyntaxKind.PlusPlusToken
                   || node.operator === ts.SyntaxKind.MinusMinusToken))
        root(node.operand);
      else if (ts.isDeleteExpression(node)) root(node.expression);
      else if (ts.isCallExpression(node)
               && ts.isPropertyAccessExpression(node.expression)
               && CHANGING.includes(node.expression.name.text))
        root(node.expression.expression);
      else if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
        const symbol = checker.getSymbolAtLocation(node.expression);
        const where = symbol && symbol.declarations && symbol.declarations[0];
        if (!where || isOurs(where.getSourceFile())) unknown = true;
      }
      node.forEachChild(walk);
    })(body);
    return { roots, unknown };
  };
  const assigned = (body, bound) =>
    writes(body).roots.some((one) => canon(one.text) === bound);
  // Everything it changes is declared inside it: outside, it did nothing.
  const contained = (statement) => {
    const made = writes(statement);
    return !made.unknown && made.roots.every((one) => {
      const symbol = checker.getSymbolAtLocation(one);
      const where = symbol && symbol.valueDeclaration;
      return where && where.getStart() >= statement.getStart()
        && where.getEnd() <= statement.getEnd();
    });
  };
  const lit = (value, type) => ({ k: "lit", value, type });
  const truth = (value) => lit(value, "boolean");
  const one = lit(1, "number");
  const known = (node) => node.k === "lit" && typeof node.value === "boolean";
  const cond = (test, yes, no) => known(test) ? (test.value ? yes : no)
    : { k: "cond", args: [test, yes, no], type: yes.type };
  const or = (a, b) => known(a) ? (a.value ? a : b) : known(b)
    ? (b.value ? b : a) : { k: "bin", op: "||", args: [a, b],
                            type: "boolean" };
  const and = (a, b) => known(a) ? (a.value ? b : a) : known(b)
    ? (b.value ? a : b) : { k: "bin", op: "&&", args: [a, b],
                            type: "boolean" };
  const not = (a) => known(a) ? truth(!a.value)
    : { k: "pre", op: "!", args: [a], type: "boolean" };

  // Control that leaves a loop early is carried as values: whether the
  // function has returned (DONE) and what (RESULT), and for each loop
  // whether it has stopped (`break`) or skips the rest of this time round
  // (`continue`). Every change after such a point is guarded by it, so the
  // loop reads as one expression and means what it did.
  const DONE = "done$", RESULT = "result$";
  let loops = [];
  let serial = 0;
  const nothing = (type) => ({ k: "opaque", text: "undefined", holes: [],
                               type });
  // A function begins having returned nothing: DONE is false, and RESULT
  // an empty value of the type it returns. A return that is not the last
  // thing on its way -- in a loop, or under an `if` the rest falls past --
  // sets them, and the end of the function says which it was.
  const begin = (returns) => {
    const plain = returns.replace(/ \| undefined/g, "");
    env.set(DONE, truth(false));
    declared.set(DONE, "boolean");
    env.set(RESULT, plain === "number" ? lit(0, "number")
      : plain === "string" ? lit("", "string")
      : plain === "boolean" ? truth(false)
      : plain.endsWith("[]") && !/[|(]/.test(plain) ? lit([], plain)
      : nothing(returns));
    declared.set(RESULT, returns);
  };
  // Falling off the end returns nothing -- unless it had returned.
  const fallen = () => {
    const done = fetch(DONE);
    const none = nothing(declared.get(RESULT));
    return known(done) ? (done.value ? fetch(RESULT) : none)
      : { k: "cond", args: [done, fetch(RESULT), none],
          type: declared.get(RESULT) };
  };
  // A return under a condition, the rest going on: carried from here.
  const returned = (when, value) => {
    const done = fetch(DONE);
    const first = known(done) && !done.value ? when : and(not(done), when);
    env.set(RESULT, { k: "cond", args: [first, value, fetch(RESULT)],
                      type: declared.get(RESULT) });
    env.set(DONE, or(fetch(DONE), copy(when)));
  };
  // A test that is not a truth is one by what the language takes it for.
  const truthy = (test) => test.type === "boolean" ? test : not(not(test));
  const halted = () => {
    let out = truth(false);
    if (!loops.length) return out;
    if (env.has(DONE)) out = or(out, fetch(DONE));
    const top = loops[loops.length - 1];
    for (const name of [top.stop, top.skip])
      if (env.has(name)) out = or(out, fetch(name));
    return out;
  };
  const guarded = (name, value) =>
    cond(halted(), fetch(name), value);

  // An assignment changes what a name is from here on.
  const counted = (expression) => (ts.isPrefixUnaryExpression(expression)
      || ts.isPostfixUnaryExpression(expression))
    ? (expression.operator === ts.SyntaxKind.PlusPlusToken ? "+"
       : expression.operator === ts.SyntaxKind.MinusMinusToken ? "-" : null)
    : null;
  const assign = (expression) => {
    const target = ts.isBinaryExpression(expression) ? expression.left
      : counted(expression) ? expression.operand : null;
    if (target && ts.isElementAccessExpression(target)) {
      // `c[k] = v`, `c[j][k] += v`, `c[k]++`: the container with that
      // element set -- at any depth, each container on the way to it one
      // with its own element set
      const levels = [];
      let base = target;
      while (ts.isElementAccessExpression(base)) {
        levels.unshift(base);
        base = base.expression;
      }
      if (ts.isIdentifier(base) && env.has(canon(base.text))) {
        const name = canon(base.text);
        let value;
        if (ts.isBinaryExpression(expression)) {
          const token = expression.operatorToken.kind;
          value = read(expression.right);
          if (token >= ts.SyntaxKind.FirstCompoundAssignment
              && token <= ts.SyntaxKind.LastCompoundAssignment)
            value = { k: "bin", op: ts.tokenToString(token).slice(0, -1),
                      args: [read(target), value], type: typeOf(target) };
          else if (token !== ts.SyntaxKind.EqualsToken)
            throw new Unread("an element changed another way");
        } else {
          value = { k: "bin", op: counted(expression),
                    args: [read(target), one], type: typeOf(target) };
        }
        const set = (level, container, type) => ({
          k: "setitem",
          args: [container, read(levels[level].argumentExpression),
                 level === levels.length - 1 ? value
                   : set(level + 1, read(levels[level]),
                         typeOf(levels[level]))],
          type });
        env.set(name, guarded(name, set(0, fetch(name),
                                        declared.get(name))));
        return;
      }
    }
    if (ts.isBinaryExpression(expression) && ts.isIdentifier(expression.left)
        && env.has(expression.left.text)) {
      const name = expression.left.text;
      if (env.get(name).k === "alias")
        throw new Unread("a second name for a list, given another");
      const token = expression.operatorToken.kind;
      const right = expression.right;
      // an empty list is of the type of the name it is given to
      const value = ts.isArrayLiteralExpression(right)
        && right.elements.length === 0
        ? lit([], declared.get(name)) : read(right);
      if (token === ts.SyntaxKind.EqualsToken) {
        env.set(name, guarded(name, value));
        return;
      }
      if (token >= ts.SyntaxKind.FirstCompoundAssignment
          && token <= ts.SyntaxKind.LastCompoundAssignment) {
        const op = ts.tokenToString(token).slice(0, -1);
        env.set(name, guarded(name, { k: "bin", op,
                                      args: [fetch(name), value],
                                      type: declared.get(name) }));
        return;
      }
    }
    if (counted(expression) && ts.isIdentifier(expression.operand)
        && env.has(expression.operand.text)) {
      const name = expression.operand.text;
      env.set(name, guarded(name, { k: "bin", op: counted(expression),
                                    args: [fetch(name), one],
                                    type: declared.get(name) }));
      return;
    }
    if (ts.isCallExpression(expression)
        && ts.isPropertyAccessExpression(expression.expression)
        && ts.isIdentifier(expression.expression.expression)
        && env.has(canon(expression.expression.expression.text))
        && CHANGING.includes(expression.expression.name.text)) {
      const name = canon(expression.expression.expression.text);
      if (expression.expression.name.text === "push"
          && expression.arguments.length === 1) {
        env.set(name, guarded(name, { k: "append",
                                      args: [fetch(name),
                                             read(expression.arguments[0])],
                                      type: declared.get(name) }));
        return;
      }
      // `c.sort()`, `s.add(x)`: the container as the call leaves it
      env.set(name, guarded(name, {
        k: "effect", member: expression.expression.name.text,
        args: [fetch(name), ...expression.arguments.map(read)],
        type: declared.get(name) }));
      return;
    }
    // Said for nothing: it changes no name and calls nothing of this
    // file's (which might) -- `xs.filter(f);` on a line of its own. What
    // it would have given is dropped, as the language drops it.
    const made = writes(expression);
    if (!made.unknown && !made.roots.length) return;
    throw new Unread(ts.isCallExpression(expression)
      ? `a call for its effect: ${expression.expression.getText()}`
      : "an expression for its effect");
  };

  // How a loop's body can leave it: return (from anywhere inside), and
  // break or continue belonging to this loop (not to one inside it).
  const exits = (statement) => {
    const out = { ret: false, brk: false, cont: false };
    (function walk(node, nested) {
      if (ts.isFunctionLike(node)) return;
      if (ts.isReturnStatement(node)) out.ret = true;
      if (!nested && ts.isBreakStatement(node)) out.brk = true;
      if (!nested && ts.isContinueStatement(node)) out.cont = true;
      const inner = nested || ts.isIterationStatement(node, false);
      node.forEachChild((child) => walk(child, inner));
    })(statement, false);
    return out;
  };
  const replaced = (node, name, by) => {
    if (Array.isArray(node)) return node.map((one) => replaced(one, name, by));
    if (!node || typeof node !== "object") return node;
    if (node.k === "id" && node.name === name) return copy(by);
    if (node.k === "arrow" && node.params.includes(name)) return node;
    const out = {};
    for (const key of Object.keys(node)) out[key] = replaced(node[key], name, by);
    return out;
  };
  // A loop that returns when it finds something: `for (x of xs) { if
  // (c) return R; }` then the rest is `xs.some(c) ? R : rest`, with `x`
  // in R the first one that does: `xs.find(c)`. The plainest reading of
  // the commonest early return; any other is read by `enter` below.
  const search = (over, element, statement, statements, at) => {
    let body = statement;
    while (ts.isBlock(body) && body.statements.length === 1)
      body = body.statements[0];
    if (!ts.isIfStatement(body) || body.elseStatement)
      throw new Unread("not a search");
    let then = body.thenStatement;
    while (ts.isBlock(then) && then.statements.length === 1)
      then = then.statements[0];
    if (!ts.isReturnStatement(then) || !then.expression)
      throw new Unread("not a search");
    const outer = env;
    env = new Map(env);
    env.delete(element);
    const test = read(body.expression);
    const value = read(then.expression);
    env = outer;
    const arrow = { k: "arrow", params: [element], body: test, type: "=>" };
    const some = { k: "call", member: "Array.some", recv: over,
                   args: [arrow], type: "boolean" };
    const find = { k: "call", member: "Array.find", recv: copy(over),
                   args: [copy(arrow)], type: over.type.replace(/\[\]$/, "") };
    const rest = run(statements, at + 1);
    if (!rest.ret) throw new Unread("no return after a loop that returns");
    return { ret: cond(some, replaced(value, element, find), rest.ret) };
  };
  const searched = (over, element, statement, statements, at) => {
    // Only where nothing may have returned yet: then `some` and `find` say
    // what the loop does whole.
    const done = env.get(DONE);
    if (loops.length || (done && !(known(done) && !done.value))) return null;
    const before = new Map(env);
    try {
      return search(over, element, statement, statements, at);
    } catch (error) {
      if (!(error instanceof Unread)) throw error;
      env = before;
      return null;
    }
  };

  // What a loop carries: the names its body (and a `for`'s step) changes,
  // and the flags of how it can be left. One is a value; more are a tuple,
  // each name its place in it (`state[1]`).
  const enter = (statement, step) => {
    const exit = exits(statement);
    if (exit.ret && !env.has(DONE))
      throw new Unread("a return inside a loop inside a callback");
    const frame = { stop: `stop$${++serial}`, skip: `skip$${serial}`,
                    state: `state${serial}`, exit };
    const carried = [...env.keys()].filter((name) => !name.includes("$")
      && (assigned(statement, name) || (step && assigned(step, name))));
    if (exit.brk || exit.ret) {
      env.set(frame.stop, truth(false));
      declared.set(frame.stop, "boolean");
      carried.push(frame.stop);
    }
    if (exit.ret) carried.push(DONE, RESULT);
    // a loop that changes nothing outside itself is nothing in what the
    // function returns
    if (!carried.length) {
      if (contained(statement)) return null;
      throw new Unread("a loop that changes what has no name here");
    }
    const type = carried.length === 1 ? declared.get(carried[0])
      : `[${carried.map((name) => declared.get(name)).join(", ")}]`;
    return { frame, carried, type };
  };
  const packed = (carried, type) => carried.length === 1
    ? fetch(carried[0])
    : { k: "tuple", args: carried.map((name) => fetch(name)), type };
  const unpack = (carried, value) => {
    let whole = value;
    if (carried.length > 1 && value.k !== "id" && frames.length) {
      // a loop's state is worked out once, however many of its places
      // are read after it
      const key = `$v${++shares}`;
      frames[frames.length - 1].defs.push({ key, value: copy(value) });
      whole = { k: "ref", key, type: value.type };
    }
    carried.forEach((name, at) => env.set(name, carried.length === 1
      ? copy(value)
      : { k: "index", args: [copy(whole), lit(at, "number")],
          type: declared.get(name) }));
  };
  // Once round the loop: the body, then a `for`'s step -- which `continue`
  // does not skip and `break` does.
  const iterate = (frame, statement, step) => {
    loops.push(frame);
    try {
      if (frame.exit.cont) {
        env.set(frame.skip, truth(false));
        declared.set(frame.skip, "boolean");
      }
      const out = block(statement);
      if (out.ret) throw new Unread("a return the loop did not see");
      env.delete(frame.skip);
      if (step) assign(step);
    } finally {
      loops.pop();
    }
  };

  // A loop over a list is a fold: `reduce` over what it goes over, its
  // body the update, the values before it the start.
  const fold = (over, element, statement) => {
    const entered = enter(statement, null);
    if (!entered) return;
    const { frame, carried, type } = entered;
    const start = packed(carried, type);
    const outer = env;
    env = new Map(env);
    const body = enterFrame();
    unpack(carried, { k: "id", name: frame.state, type });
    env.delete(element);
    iterate(frame, statement, null);
    const update = leaveFrame(body, packed(carried, type));
    env = outer;
    unpack(carried, { k: "call", member: "Array.reduce", recv: over,
                      args: [{ k: "arrow", params: [frame.state, element],
                               body: update, type: "=>" }, start],
                      type });
    env.delete(frame.stop);
  };

  // Any other loop is the language's own: while the test holds (and it has
  // not stopped), the body and a `for`'s step change what it carries.
  const loop = (test, statement, step) => {
    const entered = enter(statement, step);
    if (!entered) return;
    const { frame, carried, type } = entered;
    const start = packed(carried, type);
    const outer = env;
    env = new Map(env);
    unpack(carried, { k: "id", name: frame.state, type });
    let condition = test ? truthy(read(test)) : truth(true);
    if (carried.includes(frame.stop))
      condition = and(not(fetch(frame.stop)), condition);
    // Once the function has returned, no loop runs -- not only one that
    // returns itself: an outer return freezes this loop's changes too.
    if (env.has(DONE))
      condition = and(not(fetch(DONE)), condition);
    const body = enterFrame();
    iterate(frame, statement, step);
    const update = leaveFrame(body, packed(carried, type));
    env = outer;
    unpack(carried, { k: "while", state: frame.state,
                      args: [start, condition, update], type });
    env.delete(frame.stop);
  };
  const over = (node) => {
    const value = read(node);
    if (value.type === "string")
      return { k: "call", member: "String.split", recv: value,
               args: [lit("", "string")], type: "string[]" };
    return value;
  };
  const counting = (statement) => {
    // for (let i = A; i < B; i++) -- or <= B, or i += 1
    const list = statement.initializer;
    if (!list || !ts.isVariableDeclarationList(list)
        || list.declarations.length !== 1
        || !list.declarations[0].initializer)
      throw new Unread("a loop that does not count");
    const name = list.declarations[0].name.getText();
    const test = statement.condition;
    if (!test || !ts.isBinaryExpression(test) || test.left.getText() !== name)
      throw new Unread("a loop that does not count");
    const token = test.operatorToken.kind;
    if (token !== ts.SyntaxKind.LessThanEqualsToken
        && token !== ts.SyntaxKind.LessThanToken)
      throw new Unread("a loop that does not count up");
    const step = statement.incrementor && statement.incrementor.getText()
      .replace(/\s/g, "");
    if (![`${name}++`, `++${name}`, `${name}+=1`].includes(step))
      throw new Unread("a loop that does not count by one");
    if (assigned(statement.statement, name))
      throw new Unread("a counter changed inside its loop");
    // the bound is read once: the loop must not change what it reads
    const bound = [];
    (function walk(node) {
      if (ts.isIdentifier(node)) bound.push(node.text);
      node.forEachChild(walk);
    })(test.right);
    if (bound.some((word) => assigned(statement.statement, word)))
      throw new Unread("a loop whose bound changes");
    const from = read(list.declarations[0].initializer);
    let to = read(test.right);
    if (token === ts.SyntaxKind.LessThanEqualsToken)
      to = { k: "bin", op: "+", args: [to, one], type: "number" };
    return [{ k: "range", args: [from, to], type: "number[]" }, name];
  };
  // `let out = []` is an array of nothing yet: what it holds is what the
  // compiler says it has become where it is last read.
  const becomes = (name) => {
    const symbol = checker.getSymbolAtLocation(name);
    let found = null;
    let scope = name;
    while (scope.parent && !ts.isFunctionLike(scope)) scope = scope.parent;
    (function walk(node) {
      if (ts.isIdentifier(node) && node !== name
          && checker.getSymbolAtLocation(node) === symbol
          && !/\bany\b|never/.test(typeOf(node)))
        found = node;
      node.forEachChild(walk);
    })(scope);
    return found ? typeOf(found) : null;
  };
  const declare = (one) => {
    if (!ts.isIdentifier(one.name))
      throw new Unread("a declaration unpacked");
    if (!one.initializer) {
      let type = becomes(one.name) || typeOf(one.name);
      declared.set(one.name.text, type);
      env.set(one.name.text, { k: "opaque", text: "undefined", holes: [],
                               type });
      return;
    }
    let type = typeOf(one.name);
    if (/\bany\b|never/.test(type)) type = becomes(one.name) || type;
    declared.set(one.name.text, type);
    const init = one.initializer;
    if (ts.isIdentifier(init) && env.has(canon(init.text))
        && !["number", "string", "boolean"].includes(typeOf(init))) {
      // `let p = arr`: two names, one list. If either is then changed in
      // place, both are -- so they are read as one name; if either is
      // given another value, they part, and that is not read.
      const names = [checker.getSymbolAtLocation(one.name),
                     checker.getSymbolAtLocation(init)];
      const scope = ts.findAncestor(one, ts.isFunctionLike)
        || one.getSourceFile();
      let inPlace = false, parted = false;
      for (const root of writes(scope).roots) {
        if (!names.includes(checker.getSymbolAtLocation(root))) continue;
        const parent = root.parent;
        if ((ts.isBinaryExpression(parent) && parent.left === root)
            || ts.isPrefixUnaryExpression(parent)
            || ts.isPostfixUnaryExpression(parent)) parted = true;
        else inPlace = true;
      }
      if (inPlace) {
        if (parted)
          throw new Unread("a second name for a list, given another");
        env.set(one.name.text, { k: "alias", to: canon(init.text) });
        declared.set(one.name.text, declared.get(canon(init.text)));
        return;
      }
    }
    const value = ts.isArrayLiteralExpression(init)
      && init.elements.length === 0 ? lit([], type) : read(init);
    // what is written out (`{}`) is of the type its name says it is
    if (one.type && value.k === "opaque") value.type = type;
    env.set(one.name.text, value);
  };

  // Whether every way through these statements returns, by their shape:
  // what `run` finds by reading them, known before they are read.
  const always = (statements, from) => {
    const lands = (one) => ts.isBlock(one) ? always(one.statements, 0)
      : always([one], 0);
    for (let at = from; at < statements.length; at++) {
      const one = statements[at];
      if (ts.isReturnStatement(one)) return true;
      if (ts.isBlock(one) && always(one.statements, 0)) return true;
      if (ts.isIfStatement(one) && one.elseStatement
          && lands(one.thenStatement) && lands(one.elseStatement))
        return true;
    }
    return false;
  };
  // Statements from `at` on: {ret} when every way through returns, else
  // {} with `env` holding what the names are after them.
  const run = (statements, at) => {
    for (; at < statements.length; at++) {
      const statement = statements[at];
      if (ts.isReturnStatement(statement)) {
        const value = statement.expression ? read(statement.expression)
          : nothing(declared.get(RESULT));
        if (loops.length) {
          if (!env.has(DONE))
            throw new Unread("a return inside a loop inside a callback");
          const top = loops[loops.length - 1];
          const result = guarded(RESULT, value);
          const stop = guarded(top.stop, truth(true));
          const done = guarded(DONE, truth(true));
          env.set(RESULT, result);
          env.set(top.stop, stop);
          env.set(DONE, done);
          return {};
        }
        return { ret: env.has(DONE)
          ? cond(fetch(DONE), fetch(RESULT), value) : value };
      }
      if (ts.isBreakStatement(statement)
          || ts.isContinueStatement(statement)) {
        if (!loops.length || statement.label)
          throw new Unread("a break with nowhere to go");
        const top = loops[loops.length - 1];
        const name = ts.isBreakStatement(statement) ? top.stop : top.skip;
        env.set(name, guarded(name, truth(true)));
        return {};
      }
      if (ts.isVariableStatement(statement)) {
        statement.declarationList.declarations.forEach(declare);
        continue;
      }
      if (ts.isFunctionDeclaration(statement) && statement.name) {
        env.set(statement.name.text, read(statement));
        continue;
      }
      if (ts.isExpressionStatement(statement)) {
        assign(statement.expression);
        continue;
      }
      if (ts.isBlock(statement)) {
        const out = run(statement.statements, 0);
        if (out.ret) return out;
        continue;
      }
      if (ts.isIfStatement(statement)) {
        const test = read(statement.expression);
        const before = env;
        env = new Map(before);
        const yes = block(statement.thenStatement);
        const afterYes = env;
        env = new Map(before);
        const no = statement.elseStatement ? block(statement.elseStatement)
          : {};
        const afterNo = env;
        if (yes.ret && no.ret) return { ret: cond(test, yes.ret, no.ret) };
        if (yes.ret || no.ret) {
          // a guard: the rest is the other way through
          env = yes.ret ? afterNo : afterYes;
          if (always(statements, at + 1)) {
            const rest = run(statements, at + 1);
            if (!rest.ret) throw new Unread("no return after a guard");
            return { ret: yes.ret ? cond(test, yes.ret, rest.ret)
                                  : cond(test, rest.ret, no.ret) };
          }
          // the rest does not return on every way through (it falls out
          // of this block): the guard's return is carried, and the rest
          // read knowing it
          returned(yes.ret ? truthy(copy(test)) : not(copy(test)),
                   yes.ret || no.ret);
          return run(statements, at + 1);
        }
        env = new Map(before);
        for (const name of before.keys()) {
          const a = afterYes.get(name), b = afterNo.get(name);
          // changed the same on both ways through is changed
          if (JSON.stringify(a) === JSON.stringify(b)) env.set(name, a);
          else env.set(name, cond(copy(test), share(a), share(b)));
        }
        continue;
      }
      if (ts.isForOfStatement(statement)) {
        const list = statement.initializer;
        if (!ts.isVariableDeclarationList(list)
            || list.declarations.length !== 1
            || !ts.isIdentifier(list.declarations[0].name))
          throw new Unread("a loop over something unpacked");
        const element = list.declarations[0].name.text;
        const found = searched(over(statement.expression), element,
                               statement.statement, statements, at);
        if (found) return found;
        fold(over(statement.expression), element, statement.statement);
        continue;
      }
      if (ts.isForInStatement(statement)) {
        // the keys of what it goes over, as `Object.keys` gives them
        const list = statement.initializer;
        if (!ts.isVariableDeclarationList(list)
            || list.declarations.length !== 1
            || !ts.isIdentifier(list.declarations[0].name))
          throw new Unread("a loop over something unpacked");
        fold({ k: "call", member: "Object.keys",
               recv: { k: "opaque", text: "Object", holes: [],
                       type: "ObjectConstructor" },
               args: [read(statement.expression)], type: "string[]" },
             list.declarations[0].name.text, statement.statement);
        continue;
      }
      if (ts.isForStatement(statement)) {
        let counted = null;
        try {
          counted = counting(statement);
        } catch (error) {
          if (!(error instanceof Unread)) throw error;
        }
        if (counted) {
          const [range, name] = counted;
          const found = searched(range, name, statement.statement,
                                 statements, at);
          if (found) return found;
          fold(range, name, statement.statement);
          continue;
        }
        // for (init; test; step) body  is  init; while (test) { body; step }
        const list = statement.initializer;
        if (list && ts.isVariableDeclarationList(list))
          list.declarations.forEach(declare);
        else if (list)
          assign(list);
        loop(statement.condition, statement.statement, statement.incrementor);
        continue;
      }
      if (ts.isWhileStatement(statement)) {
        loop(statement.expression, statement.statement, null);
        continue;
      }
      throw new Unread(ts.SyntaxKind[statement.kind]);
    }
    return {};
  };
  const block = (statement) => ts.isBlock(statement)
    ? run(statement.statements, 0) : run([statement], 0);

  // A function: its parameters, and what it returns, read as one tree.
  const steps = (declaration, body) => {
    const params = declaration.parameters.map((one) => [one.name.getText(),
      one.type ? checker.typeToString(checker.getTypeFromTypeNode(one.type))
               : typeOf(one.name)]);
    const signature = checker.getSignatureFromDeclaration(declaration);
    const returns = checker.typeToString(signature.getReturnType());
    // A parameter is a variable like any other: a loop may change it.
    env = new Map(params.map(([name, type]) => [name,
                                                { k: "id", name, type }]));
    for (const [name, type] of params) declared.set(name, type);
    loops = [];
    begin(returns);
    try {
      // every name the function declares, with its type: what an edit
      // may put in the place of a name (rung 4)
      const names = params.map(([name, type]) => [name, type]);
      (function walk(node) {
        if ((ts.isVariableDeclaration(node) || ts.isParameter(node))
            && ts.isIdentifier(node.name)) {
          let type = typeOf(node.name);
          if (/\bany\b|never/.test(type)) type = becomes(node.name) || type;
          names.push([node.name.text, type]);
        }
        node.forEachChild(walk);
      })(body);
      frames = [];
      const frame = enterFrame();
      if (!ts.isBlock(body)) return { params, returns, steps: [], names,
                                      ret: leaveFrame(frame, read(body)) };
      const out = run(body.statements, 0);
      return { params, returns, steps: [], names,
               ret: leaveFrame(frame, out.ret || fallen()) };
    } catch (error) {
      if (!(error instanceof Unread)) throw error;
      return { params, returns, unread: error.message };
    }
  };
  const functions = {};
  for (const file of program.getSourceFiles()) {
    if (!isOurs(file)) continue;
    current = file.fileName;
    file.forEachChild((node) => {
      if (ts.isFunctionDeclaration(node) && node.name && node.body) {
        functions[keyOf(file, node.name.text)] = steps(node, node.body);
      } else if (ts.isVariableStatement(node)) {
        for (const one of node.declarationList.declarations) {
          const value = one.initializer;
          if (ts.isIdentifier(one.name) && value && (ts.isArrowFunction(value)
              || ts.isFunctionExpression(value)))
            functions[keyOf(file, one.name.text)] = steps(value, value.body);
        }
      }
    });
  }
  return functions;
}

function structure(source, entry) {
  const name = "read.ts";
  const program = programOf(name, PRELUDE + source);
  const checker = program.getTypeChecker();
  const file = program.getSourceFile(name);
  const uses = new Map();
  const use = (word) => uses.set(word, (uses.get(word) || 0) + 1);
  let root = null;

  const memberName = (node) => {
    const symbol = checker.getSymbolAtLocation(node.name || node);
    const declaration = symbol && symbol.declarations && symbol.declarations[0];
    if (!declaration) return null;
    const owner = declaration.parent;
    let where = owner && owner.name && owner.name.text;
    if (!where) return null;
    where = READONLY[where] || where;
    if (where === "Math" || where === "MathConstructor") where = "Math";
    return `${where.replace(/Constructor$/, "")}.${symbol.getName()}`;
  };
  const said = (node) => {
    // What an expression is, at its outermost.
    if (!node) return null;
    while (ts.isParenthesizedExpression(node) || ts.isAsExpression(node)
           || ts.isNonNullExpression(node)) node = node.expression;
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      if (ts.isPropertyAccessExpression(callee)) return memberName(callee);
      if (ts.isIdentifier(callee)) return callee.text === entry
        ? "recursion" : callee.text;
      return "call";
    }
    if (ts.isNewExpression(node)) return `new ${node.expression.getText()}`;
    if (ts.isPropertyAccessExpression(node)) return memberName(node);
    if (ts.isBinaryExpression(node)) {
      const token = ts.tokenToString(node.operatorToken.kind);
      return OPERATOR[token] || token;
    }
    if (ts.isPrefixUnaryExpression(node))
      return ts.tokenToString(node.operator);
    if (ts.isIdentifier(node)) return "a name";
    if (ts.isNumericLiteral(node) || ts.isStringLiteral(node)
        || node.kind === ts.SyntaxKind.TrueKeyword
        || node.kind === ts.SyntaxKind.FalseKeyword) return "a literal";
    return STATEMENTS[ts.SyntaxKind[node.kind]] || ts.SyntaxKind[node.kind];
  };

  let target = null;
  file.forEachChild(function find(node) {
    if (ts.isFunctionDeclaration(node) && node.name
        && node.name.text === entry) target = node;
  });
  if (!target || !target.body) return { uses: {}, root: null };
  (function walk(node, inside) {
    if (ts.isCallExpression(node) || ts.isNewExpression(node)
        || ts.isBinaryExpression(node) || ts.isPrefixUnaryExpression(node)) {
      const word = said(node);
      if (word && !(ts.isBinaryExpression(node)
                    && node.operatorToken.kind === ts.SyntaxKind.EqualsToken))
        use(word);
    } else if (ts.isPropertyAccessExpression(node)
               && !(node.parent && ts.isCallExpression(node.parent)
                    && node.parent.expression === node)) {
      const word = memberName(node);
      if (word) use(word);
    } else if (ts.isPostfixUnaryExpression(node)) {
      use(ts.tokenToString(node.operator));
    } else if (STATEMENTS[ts.SyntaxKind[node.kind]]) {
      use(STATEMENTS[ts.SyntaxKind[node.kind]]);
    }
    if (ts.isReturnStatement(node) && !inside) root = said(node.expression);
    const nested = inside || ts.isArrowFunction(node)
      || ts.isFunctionExpression(node) || ts.isFunctionDeclaration(node);
    node.forEachChild((child) => walk(child, nested));
  })(target.body, false);
  return { uses: Object.fromEntries(uses), root };
}

// What a verified program's risks are made of (`risk.py`), read off its
// syntax: the named values it keeps consistent, its decision points, and
// what a wrong one costs whoever runs it. The whole source: helpers are
// part of the program.
const MUTATING = new Set(["push", "pop", "shift", "unshift", "splice", "sort",
                          "reverse", "fill", "copyWithin", "set", "add",
                          "delete", "clear"]);
const PARSING = new Set(["parseInt", "parseFloat", "Number", "JSON.parse"]);

function qualities(source, entry) {
  const file = ts.createSourceFile("q.ts", source, ts.ScriptTarget.ES2022,
                                   true);
  const names = new Set();
  const params = new Set();
  const functions = new Map();     // name -> its declaration
  let decisions = 0, unbounded = 0, mutates = 0, partial = 0, loops = 0;
  const declare = (binding) => {
    if (!binding) return;
    if (ts.isIdentifier(binding)) names.add(binding.text);
    else binding.elements && binding.elements.forEach(
      (one) => one.name && declare(one.name));
  };
  const rootOf = (node) => {
    while (ts.isPropertyAccessExpression(node)
           || ts.isElementAccessExpression(node)
           || ts.isParenthesizedExpression(node)) node = node.expression;
    return ts.isIdentifier(node) ? node.text : null;
  };
  file.forEachChild(function top(node) {
    if (ts.isFunctionDeclaration(node) && node.name) {
      functions.set(node.name.text, node);
      if (node.name.text === entry)
        node.parameters.forEach((one) => ts.isIdentifier(one.name)
                                && params.add(one.name.text));
    } else if (ts.isVariableStatement(node)) {
      node.declarationList.declarations.forEach((one) => {
        if (one.initializer && ts.isIdentifier(one.name)
            && (ts.isArrowFunction(one.initializer)
                || ts.isFunctionExpression(one.initializer)))
          functions.set(one.name.text, one.initializer);
      });
    }
  });
  (function walk(node, owner) {
    if (ts.isFunctionDeclaration(node) || ts.isArrowFunction(node)
        || ts.isFunctionExpression(node) || ts.isMethodDeclaration(node)) {
      // a helper is a named value the program keeps; the function asked
      // for is not one of its own
      if (node.name && ts.isIdentifier(node.name) && node.name.text !== entry)
        names.add(node.name.text);
      node.parameters.forEach((one) => declare(one.name));
      const name = node.name && ts.isIdentifier(node.name) ? node.name.text
        : (node.parent && ts.isVariableDeclaration(node.parent)
           && ts.isIdentifier(node.parent.name) ? node.parent.name.text
           : owner);
      node.forEachChild((child) => walk(child, name));
      return;
    }
    if (ts.isVariableDeclaration(node)) declare(node.name);
    if (ts.isIfStatement(node) || ts.isConditionalExpression(node)
        || ts.isCaseClause(node) || ts.isBreakStatement(node)
        || ts.isContinueStatement(node)) decisions++;
    if (ts.isBinaryExpression(node)) {
      const kind = node.operatorToken.kind;
      if (kind === ts.SyntaxKind.AmpersandAmpersandToken
          || kind === ts.SyntaxKind.BarBarToken
          || kind === ts.SyntaxKind.QuestionQuestionToken) decisions++;
      if (kind === ts.SyntaxKind.SlashToken
          || kind === ts.SyntaxKind.PercentToken) partial++;
      if (kind >= ts.SyntaxKind.FirstAssignment
          && kind <= ts.SyntaxKind.LastAssignment
          && !ts.isIdentifier(node.left) && params.has(rootOf(node.left)))
        mutates++;
    }
    if (ts.isWhileStatement(node) || ts.isDoStatement(node)) unbounded++;
    if (ts.isForStatement(node) || ts.isForOfStatement(node)
        || ts.isForInStatement(node) || ts.isWhileStatement(node)
        || ts.isDoStatement(node)) loops++;
    if (ts.isElementAccessExpression(node) || ts.isThrowStatement(node)
        || ts.isNonNullExpression(node)) partial++;
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      if (ts.isIdentifier(callee)) {
        if (callee.text === owner && functions.has(owner)) unbounded++;
        if (PARSING.has(callee.text)) partial++;
      } else if (ts.isPropertyAccessExpression(callee)) {
        const said = callee.getText();
        if (PARSING.has(said)) partial++;
        if (MUTATING.has(callee.name.text)
            && params.has(rootOf(callee.expression))) mutates++;
      }
    }
    node.forEachChild((child) => walk(child, owner));
  })(file, null);
  return { names: names.size, decisions, unbounded, mutates, partial, loops,
           found: functions.has(entry) };
}

// A project's outline, off its syntax (v698): each file's functions --
// declared, bound to a name, a class's methods -- with their lines, their
// parameters and result as written, what each calls by name; and what each
// file imports from where. No type checker: it is read as fast as it is
// parsed, and a file that does not compile is still outlined.
function outline(files) {
  const out = {};
  for (const [file, text] of Object.entries(files)) {
    const kind = /\.[jt]sx$/.test(file) ? ts.ScriptKind.TSX
      : /\.(js|mjs|cjs)$/.test(file) ? ts.ScriptKind.JS : ts.ScriptKind.TS;
    const sf = ts.createSourceFile(file, text, ts.ScriptTarget.ES2022, true,
                                   kind);
    const line = (pos) => sf.getLineAndCharacterOfPosition(pos).line + 1;
    const functions = [];
    const imports = [];
    const exported = new Set();
    const isExported = (node) => !!(ts.getCombinedModifierFlags
      && (ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Export));
    const record = (name, node, kindName, owner, exportedHere) => {
      const calls = new Set();
      const body = node.body || node;
      (function walk(n) {
        if (ts.isCallExpression(n) || ts.isNewExpression(n)) {
          const callee = n.expression;
          if (ts.isIdentifier(callee)) calls.add(callee.text);
          else if (ts.isPropertyAccessExpression(callee)) {
            const said = callee.getText(sf);
            calls.add(said.length <= 60 ? said : callee.name.text);
          }
        }
        n.forEachChild(walk);
      })(body);
      functions.push({
        name, kind: kindName, class: owner || null,
        exported: exportedHere,
        start: line(node.getStart(sf)), end: line(node.end),
        params: (node.parameters || []).map((p) => [
          p.name.getText(sf), p.type ? p.type.getText(sf) : null]),
        returns: node.type ? node.type.getText(sf) : null,
        async: !!(node.modifiers || []).some(
          (m) => m.kind === ts.SyntaxKind.AsyncKeyword),
        calls: [...calls].sort(),
      });
    };
    sf.forEachChild(function top(node) {
      if (ts.isFunctionDeclaration(node) && node.name) {
        record(node.name.text, node, "function", null, isExported(node));
      } else if (ts.isVariableStatement(node)) {
        for (const one of node.declarationList.declarations) {
          const init = one.initializer;
          if (init && ts.isIdentifier(one.name)
              && (ts.isArrowFunction(init) || ts.isFunctionExpression(init)))
            record(one.name.text, init, "function", null, isExported(node));
        }
      } else if (ts.isClassDeclaration(node) && node.name) {
        for (const member of node.members) {
          if ((ts.isMethodDeclaration(member) || ts.isConstructorDeclaration(member))
              && member.body) {
            const name = member.name ? member.name.getText(sf) : "constructor";
            record(`${node.name.text}.${name}`, member, "method",
                   node.name.text, isExported(node));
          }
        }
      } else if (ts.isImportDeclaration(node)) {
        const names = [];
        const clause = node.importClause;
        if (clause && clause.name) names.push(clause.name.text);
        if (clause && clause.namedBindings) {
          if (ts.isNamespaceImport(clause.namedBindings))
            names.push(`* as ${clause.namedBindings.name.text}`);
          else clause.namedBindings.elements.forEach(
            (e) => names.push(e.name.text));
        }
        imports.push({ from: node.moduleSpecifier.text, names,
                       line: line(node.getStart(sf)) });
      } else if (ts.isExportDeclaration(node) && node.exportClause
                 && ts.isNamedExports(node.exportClause)) {
        node.exportClause.elements.forEach((e) => exported.add(e.name.text));
      }
    });
    for (const one of functions) if (exported.has(one.name)) one.exported = true;
    out[file] = { functions, imports, lines: sf.getLineStarts().length };
  }
  return out;
}

const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    return;
  }
  const reply = { id: request.id };
  try {
    if (request.op === "check") {
      reply.errors = check(request.source);
      reply.ok = reply.errors.length === 0;
    } else if (request.op === "run") {
      reply.outputs = run(request.source, request.entry, request.cases,
                          request.timeout || 200);
      reply.ok = true;
    } else if (request.op === "values") {
      reply.values = values(request.params, request.cases,
                            request.expressions, request.timeout || 50,
                            request.prelude || "");
      reply.ok = true;
    } else if (request.op === "signatures") {
      reply.signatures = signatures(request.receivers || [],
                                    request.globals || []);
      reply.ok = true;
    } else if (request.op === "tree") {
      reply.tree = tree(request.source, request.files);
      reply.ok = true;
    } else if (request.op === "diagnose") {
      reply.errors = diagnose(request.files);
      reply.ok = true;
    } else if (request.op === "project") {
      project(request.files, request.main, request.timeout || 2000);
      reply.ok = true;
    } else if (request.op === "structure") {
      Object.assign(reply, structure(request.source, request.entry));
      reply.ok = true;
    } else if (request.op === "outline") {
      reply.outline = outline(request.files);
      reply.ok = true;
    } else if (request.op === "qualities") {
      reply.qualities = qualities(request.source, request.entry);
      reply.ok = true;
    } else if (request.op === "tests") {
      tests(request.source, request.timeout || 2000);
      reply.ok = true;
    } else {
      reply.ok = false;
      reply.error = `unknown op ${request.op}`;
    }
  } catch (error) {
    reply.ok = false;
    reply.error = String(error && error.message || error);
  }
  process.stdout.write(JSON.stringify(reply) + "\n");
});
