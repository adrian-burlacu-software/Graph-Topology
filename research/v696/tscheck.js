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
      return reported(value);
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
  // The helpers the expressions call, declared first (rung 3).
  if (prelude) vm.runInContext(transpile(prelude), context, { timeout });
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

function programOf(name, text) {
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
  return ts.createProgram([name], OPTIONS, host);
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
function tree(source) {
  const name = "tree.ts";
  const program = programOf(name, PRELUDE + source);
  const checker = program.getTypeChecker();
  const file = program.getSourceFile(name);
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
    for (const def of [...frame.defs].reverse())
      out = { k: "let", key: def.key, value: def.value, body: out,
              type: out.type };
    return out;
  };
  const fetch = (name) => share(env.get(name));
  // A value put inside another -- read by its name, or kept as the other
  // side of a merged `if` -- is shared the same way: never copied whole
  // more than once.
  const share = (value) => {
    const text = JSON.stringify(value);
    if (!value || typeof value !== "object"
        || (text.match(/"k":/g) || []).length <= SHARED)
      return JSON.parse(text);
    if (!value.$share && !value.$read) {
      // read once: nothing to share yet -- it is where it is read. A
      // second read binds it (so no value is ever more than twice over).
      Object.defineProperty(value, "$read", { value: true,
                                              enumerable: false });
      return JSON.parse(text);
    }
    if (!value.$share) {
      // made in the innermost frame that did not inherit it
      let home = null;
      for (let at = frames.length - 1; at >= 0; at--) {
        if (!frames[at].inherited.has(value)) {
          home = frames[at];
          break;
        }
      }
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
  const span = (node) => [node.getStart() - at, node.getEnd() - at];
  const read = (node) => {
    const out = readNode(node);
    if (out && typeof out === "object" && !out.span) {
      out.span = span(node);
      let inner = node;
      while (ts.isParenthesizedExpression(inner) || ts.isAsExpression(inner)
             || ts.isNonNullExpression(inner)) inner = inner.expression;
      if (ts.isBinaryExpression(inner))
        out.opspan = span(inner.operatorToken);
      else if (ts.isPrefixUnaryExpression(inner))
        out.opspan = [inner.getStart() - at, inner.operand.getStart() - at];
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
        const symbol = checker.getSymbolAtLocation(callee);
        const where = symbol && symbol.declarations && symbol.declarations[0];
        // what the language itself declares (`isNaN`, `Number`), as against
        // a helper of this file
        const global = !!(where && where.getSourceFile().fileName !== name);
        return { k: "call", member: callee.text, recv: null, args, type,
                 global };
      }
    } else if (ts.isPropertyAccessExpression(node)) {
      return { k: "prop", member: memberName(node),
               recv: read(node.expression), type };
    } else if (ts.isBinaryExpression(node)) {
      const token = ts.tokenToString(node.operatorToken.kind);
      return { k: "bin", op: OPERATOR[token] || token,
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
      env = new Map(env);
      for (const one of params) env.delete(one);
      env.delete(DONE);
      env.delete(RESULT);
      loops = [];
      const frame = enterFrame();
      try {
        if (!ts.isBlock(node.body))
          return { k: "arrow", params,
                   body: leaveFrame(frame, read(node.body)), type };
        const out = run(node.body.statements, 0);
        if (out.ret)
          return { k: "arrow", params, body: leaveFrame(frame, out.ret),
                   type };
      } catch (error) {
        if (!(error instanceof Unread)) throw error;
      } finally {
        if (frames.includes(frame)) frames.splice(frames.indexOf(frame), 1);
        env = outer;
        loops = outerLoops;
      }
      return opaque(node, type);
    } else if (ts.isElementAccessExpression(node)) {
      return { k: "index", args: [read(node.expression),
                                  read(node.argumentExpression)], type };
    } else if (ts.isIdentifier(node)) {
      if (env.has(node.text)) return fetch(node.text);
      if (node.text === "Infinity")
        return { k: "lit", value: "Infinity", type: "number" };
      const symbol = checker.getSymbolAtLocation(node);
      const where = symbol && symbol.declarations && symbol.declarations[0];
      if (where && where.getSourceFile().fileName !== name)
        // what the language declares, used as a value: `String`, `NaN`
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
            && where.getSourceFile().fileName === name) {
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

  const assigned = (body, bound) => {
    let found = false;
    (function walk(node) {
      if ((ts.isBinaryExpression(node) && node.operatorToken.kind
           >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind
           <= ts.SyntaxKind.LastAssignment && ts.isIdentifier(node.left)
           && node.left.text === bound)
          || ((ts.isPrefixUnaryExpression(node)
               || ts.isPostfixUnaryExpression(node))
              && ts.isIdentifier(node.operand) && node.operand.text === bound
              && (node.operator === ts.SyntaxKind.PlusPlusToken
                  || node.operator === ts.SyntaxKind.MinusMinusToken))
          // a list changed in place is a value changed
          || (ts.isCallExpression(node)
              && ts.isPropertyAccessExpression(node.expression)
              && ts.isIdentifier(node.expression.expression)
              && node.expression.expression.text === bound
              && CHANGING.includes(node.expression.name.text))
          // an element set is the container changed
          || (ts.isBinaryExpression(node) && node.operatorToken.kind
              >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind
              <= ts.SyntaxKind.LastAssignment
              && ts.isElementAccessExpression(node.left)
              && ts.isIdentifier(node.left.expression)
              && node.left.expression.text === bound))
        found = true;
      node.forEachChild(walk);
    })(body);
    return found;
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
  const assign = (expression) => {
    if (ts.isBinaryExpression(expression)
        && ts.isElementAccessExpression(expression.left)
        && ts.isIdentifier(expression.left.expression)
        && env.has(expression.left.expression.text)) {
      // `c[k] = v`: the container with that element set
      const name = expression.left.expression.text;
      const token = expression.operatorToken.kind;
      const key = read(expression.left.argumentExpression);
      let value = read(expression.right);
      if (token >= ts.SyntaxKind.FirstCompoundAssignment
          && token <= ts.SyntaxKind.LastCompoundAssignment)
        value = { k: "bin", op: ts.tokenToString(token).slice(0, -1),
                  args: [read(expression.left), value],
                  type: typeOf(expression.left) };
      else if (token !== ts.SyntaxKind.EqualsToken)
        throw new Unread("an element changed another way");
      env.set(name, guarded(name, { k: "setitem",
                                    args: [fetch(name), key, value],
                                    type: declared.get(name) }));
      return;
    }
    if (ts.isBinaryExpression(expression) && ts.isIdentifier(expression.left)
        && env.has(expression.left.text)) {
      const name = expression.left.text;
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
    if ((ts.isPrefixUnaryExpression(expression)
         || ts.isPostfixUnaryExpression(expression))
        && ts.isIdentifier(expression.operand)
        && env.has(expression.operand.text)) {
      const name = expression.operand.text;
      const op = expression.operator === ts.SyntaxKind.PlusPlusToken ? "+"
        : expression.operator === ts.SyntaxKind.MinusMinusToken ? "-" : null;
      if (op) {
        env.set(name, guarded(name, { k: "bin", op,
                                      args: [fetch(name), one],
                                      type: declared.get(name) }));
        return;
      }
    }
    if (ts.isCallExpression(expression)
        && ts.isPropertyAccessExpression(expression.expression)
        && expression.expression.name.text === "push"
        && ts.isIdentifier(expression.expression.expression)
        && env.has(expression.expression.expression.text)
        && expression.arguments.length === 1) {
      const name = expression.expression.expression.text;
      env.set(name, guarded(name, { k: "append",
                                    args: [fetch(name),
                                           read(expression.arguments[0])],
                                    type: declared.get(name) }));
      return;
    }
    if (ts.isCallExpression(expression)
        && ts.isPropertyAccessExpression(expression.expression)
        && ts.isIdentifier(expression.expression.expression)
        && env.has(expression.expression.expression.text)
        && CHANGING.includes(expression.expression.name.text)) {
      // `c.sort()`, `s.add(x)`: the container as the call leaves it
      const name = expression.expression.expression.text;
      env.set(name, guarded(name, {
        k: "effect", member: expression.expression.name.text,
        args: [fetch(name), ...expression.arguments.map(read)],
        type: declared.get(name) }));
      return;
    }
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
    if (!carried.length) throw new Unread("a loop that changes nothing");
    const type = carried.length === 1 ? declared.get(carried[0])
      : `[${carried.map((name) => declared.get(name)).join(", ")}]`;
    return { frame, carried, type };
  };
  const packed = (carried, type) => carried.length === 1
    ? fetch(carried[0])
    : { k: "tuple", args: carried.map((name) => fetch(name)), type };
  const unpack = (carried, value) => {
    carried.forEach((name, at) => env.set(name, carried.length === 1
      ? copy(value)
      : { k: "index", args: [copy(value), lit(at, "number")],
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
    const { frame, carried, type } = enter(statement, null);
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
    const { frame, carried, type } = enter(statement, step);
    const start = packed(carried, type);
    const outer = env;
    env = new Map(env);
    unpack(carried, { k: "id", name: frame.state, type });
    let condition = test ? read(test) : truth(true);
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
    env.set(one.name.text, ts.isArrayLiteralExpression(init)
      && init.elements.length === 0
      ? lit([], type) : read(init));
  };

  // Statements from `at` on: {ret} when every way through returns, else
  // {} with `env` holding what the names are after them.
  const run = (statements, at) => {
    for (; at < statements.length; at++) {
      const statement = statements[at];
      if (ts.isReturnStatement(statement)) {
        if (!statement.expression) throw new Unread("returns nothing");
        const value = read(statement.expression);
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
          const rest = run(statements, at + 1);
          if (!rest.ret) throw new Unread("no return after a guard");
          return { ret: yes.ret ? cond(test, yes.ret, rest.ret)
                                : cond(test, rest.ret, no.ret) };
        }
        env = new Map(before);
        for (const name of before.keys()) {
          const a = afterYes.get(name), b = afterNo.get(name);
          if (JSON.stringify(a) !== JSON.stringify(b))
            env.set(name, cond(copy(test), share(a), share(b)));
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
    let early = false;
    (function walk(node, looping) {
      if (node !== declaration && ts.isFunctionLike(node)) return;
      if (looping && ts.isReturnStatement(node)) early = true;
      const inner = looping || ts.isIterationStatement(node, false);
      node.forEachChild((child) => walk(child, inner));
    })(body, false);
    if (early) {
      // Before anything has returned, what it would return is nothing yet:
      // an empty value of the type it returns.
      const plain = returns.replace(/ \| undefined/g, "");
      const empty = plain === "number" ? lit(0, "number")
        : plain === "string" ? lit("", "string")
        : plain === "boolean" ? truth(false)
        : plain.endsWith("[]") ? lit([], plain)
        : { k: "opaque", text: "undefined", holes: [], type: returns };
      if (!empty)
        return { params, returns, unread: `a loop that returns ${returns}` };
      env.set(DONE, truth(false));
      declared.set(DONE, "boolean");
      env.set(RESULT, empty);
      declared.set(RESULT, returns);
    }
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
      if (!out.ret) return { params, returns, unread: "no return" };
      return { params, returns, steps: [], names,
               ret: leaveFrame(frame, out.ret) };
    } catch (error) {
      if (!(error instanceof Unread)) throw error;
      return { params, returns, unread: error.message };
    }
  };
  const functions = {};
  file.forEachChild((node) => {
    if (ts.isFunctionDeclaration(node) && node.name && node.body) {
      functions[node.name.text] = steps(node, node.body);
    } else if (ts.isVariableStatement(node)) {
      for (const one of node.declarationList.declarations) {
        const value = one.initializer;
        if (ts.isIdentifier(one.name) && value && (ts.isArrowFunction(value)
            || ts.isFunctionExpression(value)))
          functions[one.name.text] = steps(value, value.body);
      }
    }
  });
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
      reply.tree = tree(request.source);
      reply.ok = true;
    } else if (request.op === "structure") {
      Object.assign(reply, structure(request.source, request.entry));
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
