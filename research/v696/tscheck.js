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

function sandbox() {
  const context = { require, console: { log: () => {} }, module: {},
                    exports: {} };
  return vm.createContext(context);
}

function run(source, entry, cases, timeout) {
  const context = sandbox();
  vm.runInContext(transpile(source) + `\n;globalThis.__entry = ${entry};`,
                  context, { timeout });
  return cases.map((args) => {
    try {
      context.__args = args;
      const value = vm.runInContext("__entry(...__args)", context,
                                    { timeout });
      return { value: value === undefined ? null : value };
    } catch (error) {
      return { error: String(error && error.message || error) };
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
      if (calls.length === 0) {
        out.push({ receiver: receivers[index], name: property.getName(),
                   property: true, params: [], deprecated,
                   returns: checker.typeToString(member), generic: false });
        continue;
      }
      calls.forEach((signature) => out.push(Object.assign(
        { receiver: receivers[index], name: property.getName(),
          property: false, deprecated }, describe(signature, declaration.name))));
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
            : `Math.${property.getName()}`, property: false },
        describe(signature, source))));
    }
  }
  return out;
}

// Many expressions at once over the same parameters and cases: each
// expression's value on each case, or its error. The candidates of a
// search are evaluated here in batches, so what code does is always what
// Node says it does, and never a copy of it.
function values(params, cases, expressions, timeout) {
  const context = sandbox();
  context.__cases = cases;
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
        row.push({ value: value === undefined ? null
          : (typeof value === "number" && !isFinite(value) ? String(value) : value) });
      } catch (error) {
        row.push({ error: String(error && error.message || error).slice(0, 80) });
      }
    }
    out.push(row);
  }
  return out;
}

function tests(source, timeout) {
  vm.runInContext(transpile(source), sandbox(), { timeout });
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
                            request.expressions, request.timeout || 50);
      reply.ok = true;
    } else if (request.op === "signatures") {
      reply.signatures = signatures(request.receivers || [],
                                    request.globals || []);
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
