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
  const copy = (value) => JSON.parse(JSON.stringify(value));
  const read = (node) => {
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
      if (ts.isIdentifier(callee))
        return { k: "call", member: callee.text, recv: null, args, type };
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
    } else if (ts.isArrowFunction(node) || ts.isFunctionExpression(node)) {
      const params = node.parameters.map((one) => one.name.getText());
      const outer = env;
      env = new Map(env);
      for (const one of params) env.delete(one);
      try {
        if (!ts.isBlock(node.body))
          return { k: "arrow", params, body: read(node.body), type };
        const out = run(node.body.statements, 0);
        if (!out.ret) return { k: "other", text: node.getText(), type };
        return { k: "arrow", params, body: out.ret, type };
      } catch (error) {
        if (!(error instanceof Unread)) throw error;
        return { k: "other", text: error.message, type };
      } finally {
        env = outer;
      }
    } else if (ts.isElementAccessExpression(node)) {
      return { k: "index", args: [read(node.expression),
                                  read(node.argumentExpression)], type };
    } else if (ts.isIdentifier(node)) {
      if (env.has(node.text)) return copy(env.get(node.text));
      if (node.text === "Infinity")
        return { k: "lit", value: "Infinity", type: "number" };
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
    return { k: "other", text: node.getText(), type };
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
              && ["push", "pop", "shift", "unshift", "splice", "sort",
                  "reverse", "fill", "copyWithin"].includes(
                node.expression.name.text)))
        found = true;
      node.forEachChild(walk);
    })(body);
    return found;
  };
  const cond =(test, yes, no) => ({ k: "cond", args: [test, yes, no],
                                      type: yes.type });
  const one = { k: "lit", value: 1, type: "number" };

  // An assignment changes what a name is from here on.
  const assign = (expression) => {
    if (ts.isBinaryExpression(expression) && ts.isIdentifier(expression.left)
        && env.has(expression.left.text)) {
      const name = expression.left.text;
      const token = expression.operatorToken.kind;
      const value = read(expression.right);
      if (token === ts.SyntaxKind.EqualsToken) {
        env.set(name, value);
        return;
      }
      if (token >= ts.SyntaxKind.FirstCompoundAssignment
          && token <= ts.SyntaxKind.LastCompoundAssignment) {
        const op = ts.tokenToString(token).slice(0, -1);
        env.set(name, { k: "bin", op, args: [copy(env.get(name)), value],
                        type: declared.get(name) });
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
        env.set(name, { k: "bin", op, args: [copy(env.get(name)), one],
                        type: declared.get(name) });
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
      env.set(name, { k: "append", args: [copy(env.get(name)),
                                          read(expression.arguments[0])],
                      type: declared.get(name) });
      return;
    }
    throw new Unread(ts.isCallExpression(expression)
      ? `a call for its effect: ${expression.expression.getText()}`
      : "an expression for its effect");
  };

  const returns = (node) => {
    let found = false;
    (function walk(one) {
      if (ts.isReturnStatement(one)) found = true;
      if (!ts.isFunctionLike(one)) one.forEachChild(walk);
    })(node);
    return found;
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
  // in R the first one that does: `xs.find(c)`.
  const search = (over, element, statement, statements, at) => {
    let body = statement;
    while (ts.isBlock(body) && body.statements.length === 1)
      body = body.statements[0];
    if (!ts.isIfStatement(body) || body.elseStatement)
      throw new Unread("a loop that returns other than when it finds");
    let then = body.thenStatement;
    while (ts.isBlock(then) && then.statements.length === 1)
      then = then.statements[0];
    if (!ts.isReturnStatement(then) || !then.expression)
      throw new Unread("a loop that returns other than when it finds");
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

  // A loop carrying one value is a fold: `reduce` over what it goes over,
  // its body the update, the value before it the initial one.
  const fold = (over, element, statement) => {
    const carried = [...env.keys()].filter((name) =>
      assigned(statement, name));
    if (carried.length !== 1)
      throw new Unread(`a loop carrying ${carried.length} values`);
    const acc = carried[0];
    const outer = env;
    env = new Map(env);
    env.set(acc, { k: "id", name: acc, type: declared.get(acc) });
    env.delete(element);
    let out;
    try {
      out = block(statement);
    } finally {
      const update = env.get(acc);
      env = outer;
      if (out && out.ret) throw new Unread("a return inside a loop");
      env.set(acc, { k: "call", member: "Array.reduce", recv: over,
                     args: [{ k: "arrow", params: [acc, element],
                              body: update, type: "=>" },
                            copy(outer.get(acc))],
                     type: declared.get(acc) });
    }
  };
  const over = (node) => {
    const value = read(node);
    if (value.type === "string")
      return { k: "call", member: "String.split", recv: value,
               args: [{ k: "lit", value: "", type: "string" }],
               type: "string[]" };
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
    const from = read(list.declarations[0].initializer);
    const test = statement.condition;
    if (!test || !ts.isBinaryExpression(test) || test.left.getText() !== name)
      throw new Unread("a loop that does not count");
    let to = read(test.right);
    const token = test.operatorToken.kind;
    if (token === ts.SyntaxKind.LessThanEqualsToken)
      to = { k: "bin", op: "+", args: [to, one], type: "number" };
    else if (token !== ts.SyntaxKind.LessThanToken)
      throw new Unread("a loop that does not count up");
    const step = statement.incrementor && statement.incrementor.getText()
      .replace(/\s/g, "");
    if (![`${name}++`, `++${name}`, `${name}+=1`].includes(step))
      throw new Unread("a loop that does not count by one");
    if (assigned(statement.statement, name))
      throw new Unread("a counter changed inside its loop");
    return [{ k: "range", args: [from, to], type: "number[]" }, name];
  };

  // Statements from `at` on: {ret} when every way through returns, else
  // {} with `env` holding what the names are after them.
  const run = (statements, at) => {
    for (; at < statements.length; at++) {
      const statement = statements[at];
      if (ts.isReturnStatement(statement)) {
        if (!statement.expression) throw new Unread("returns nothing");
        return { ret: read(statement.expression) };
      }
      if (ts.isVariableStatement(statement)) {
        for (const one of statement.declarationList.declarations) {
          if (!one.initializer || !ts.isIdentifier(one.name))
            throw new Unread("a declaration without a value");
          declared.set(one.name.text, typeOf(one.name));
          const init = one.initializer;
          env.set(one.name.text, ts.isArrayLiteralExpression(init)
            && init.elements.length === 0
            ? { k: "lit", value: [], type: typeOf(one.name) }
            : read(init));
        }
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
            env.set(name, cond(copy(test), a, b));
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
        if (returns(statement.statement))
          return search(over(statement.expression), element,
                        statement.statement, statements, at);
        fold(over(statement.expression), element, statement.statement);
        continue;
      }
      if (ts.isForStatement(statement)) {
        const [range, name] = counting(statement);
        if (returns(statement.statement))
          return search(range, name, statement.statement, statements, at);
        fold(range, name, statement.statement);
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
    env = new Map();
    try {
      if (!ts.isBlock(body)) return { params, returns, steps: [],
                                      ret: read(body) };
      const out = run(body.statements, 0);
      if (!out.ret) return { params, returns, unread: "no return" };
      return { params, returns, steps: [], ret: out.ret };
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
