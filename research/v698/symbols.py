"""Every name a Python project defines, and every place that uses it (v698).

The outline (`v696/pycheck.outline`) holds a project's functions and the
names each calls -- an attribute call by its attribute alone (`load_books`
of `B.load_books()`), so a call through a module (`from library import
books as B`) was followed nowhere: "Nothing in the project calls
load_books", where main and handle do. A constant (`LOAN_DAYS`) or a class
(`Book`) was no subject at all ("it is not a function or file of the
project").

Here each file is read whole, by its syntax tree:

- what it **defines**: functions, classes (their fields and methods),
  methods, constants -- each with its file, lines, and what is said of it
  (a docstring, or a `#:` comment above);
- what **uses** each: a call, a read, an import, a type -- from which
  function (or the module itself), followed through imports and module
  aliases to the file that defines it;
- what a function **answers to**: the strings it compares a parameter to
  (`if argv[0] == "books"`), and the keys of a table of functions it looks
  up (`ROUTES = {"/plants": _plants}`) -- its commands, its routes.

Looked up, never guessed: a use is a use only where the name is followed
to its definition.
"""
from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str               # `Book`, `Book.title`, `Order.total`, `LOAN_DAYS`
    kind: str               # function | method | class | field | constant
    file: str
    start: int
    end: int
    doc: str = ""
    value: str = ""         # a constant's or a field's default, as written
    owner: str = ""         # a method's or a field's class
    params: list = field(default_factory=list)

    def json(self) -> dict:
        out = {"name": self.name, "kind": self.kind, "file": self.file,
               "start": self.start, "end": self.end}
        for key in ("doc", "value", "owner"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.params:
            out["params"] = self.params
        return out


@dataclass
class Use:
    symbol: str             # the defining file#name it is of
    file: str
    line: int
    by: str                 # the function (or `Class.method`) it is in; ""
                            # at the top of the module
    how: str                # call | read | import | type


def _doc_above(lines: list, line: int) -> str:
    """The `#:` comment lines just above `line` (1-based)."""
    found = []
    at = line - 2
    while at >= 0 and lines[at].strip().startswith("#"):
        found.insert(0, lines[at].strip().lstrip("#:").strip())
        at -= 1
    return " ".join(one for one in found if one)


def _module_of(files: set, path: str, module: str) -> str | None:
    """The project's file an import names, as `Project._python_module`."""
    dots = len(module) - len(module.lstrip("."))
    parts = [one for one in module.lstrip(".").split(".") if one]
    if dots:
        base = posixpath.dirname(path)
        for _ in range(dots - 1):
            base = posixpath.dirname(base)
        starts = [base]
    else:
        top = path.split("/")[0] if "/" in path else ""
        starts = ["", top, posixpath.dirname(path)]
    for start in starts:
        stem = posixpath.join(start, *parts) if parts else start
        for candidate in (stem + ".py", posixpath.join(stem, "__init__.py")):
            candidate = candidate.lstrip("/")
            if candidate in files:
                return candidate
    return None


class Index:
    """A Python project's names: `Index(files)` with {path: text}."""

    def __init__(self, files: dict) -> None:
        self.files = {path: text for path, text in files.items()
                      if path.endswith(".py") and text is not None}
        self.symbols: dict = {}         # file#name -> Symbol
        self.uses: list = []
        self.cases: dict = {}           # file#function -> [strings]
        #: file#function -> what it calls of the language and libraries:
        #: a builtin, a name imported from outside the project, `module.f`
        self.outside: dict = {}
        self._trees = {}
        for path, text in sorted(self.files.items()):
            try:
                self._trees[path] = ast.parse(text)
            except SyntaxError:
                continue
            self._define(path, self._trees[path], text.splitlines())
        for path, tree in self._trees.items():
            self._use(path, tree)

    # -- what each file defines -------------------------------------------------
    def _define(self, path: str, tree, lines: list) -> None:
        def put(one: Symbol) -> None:
            self.symbols[f"{path}#{one.name}"] = one

        def function(node, owner: str = "") -> None:
            args = [one.arg for one in node.args.args
                    if not (owner and one.arg in ("self", "cls"))]
            put(Symbol(f"{owner}.{node.name}" if owner else node.name,
                       "method" if owner else "function", path, node.lineno,
                       node.end_lineno or node.lineno,
                       ast.get_docstring(node) or _doc_above(lines,
                                                             node.lineno),
                       owner=owner, params=args))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function(node)
            elif isinstance(node, ast.ClassDef):
                put(Symbol(node.name, "class", path, node.lineno,
                           node.end_lineno or node.lineno,
                           ast.get_docstring(node) or _doc_above(
                               lines, node.lineno)))
                for inner in node.body:
                    if isinstance(inner, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                        function(inner, node.name)
                        if inner.name == "__init__":
                            # fields set in the constructor: self.x = ...
                            for one in ast.walk(inner):
                                if isinstance(one, ast.Assign):
                                    for target in one.targets:
                                        self._field(put, path, node.name,
                                                    target, one, lines)
                    elif isinstance(inner, ast.AnnAssign) and isinstance(
                            inner.target, ast.Name):
                        put(Symbol(f"{node.name}.{inner.target.id}", "field",
                                   path, inner.lineno,
                                   inner.end_lineno or inner.lineno,
                                   _doc_above(lines, inner.lineno),
                                   ast.unparse(inner.value)
                                   if inner.value is not None else "",
                                   owner=node.name))
                    elif isinstance(inner, ast.Assign):
                        for target in inner.targets:
                            if isinstance(target, ast.Name):
                                put(Symbol(f"{node.name}.{target.id}",
                                           "field", path, inner.lineno,
                                           inner.end_lineno or inner.lineno,
                                           _doc_above(lines, inner.lineno),
                                           ast.unparse(inner.value),
                                           owner=node.name))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and node.value is not None:
                        put(Symbol(target.id, "constant", path, node.lineno,
                                   node.end_lineno or node.lineno,
                                   _doc_above(lines, node.lineno),
                                   ast.unparse(node.value)))

    def _field(self, put, path: str, owner: str, target, node,
               lines: list) -> None:
        if isinstance(target, ast.Attribute) and isinstance(
                target.value, ast.Name) and target.value.id == "self":
            key = f"{path}#{owner}.{target.attr}"
            if key not in self.symbols:
                put(Symbol(f"{owner}.{target.attr}", "field", path,
                           node.lineno, node.end_lineno or node.lineno,
                           _doc_above(lines, node.lineno),
                           ast.unparse(node.value), owner=owner))

    # -- what uses what ---------------------------------------------------------
    def _bound(self, path: str, tree) -> tuple:
        """What the module's names are: {name: file#symbol} of what it
        imports by name or defines, and {alias: file} of the modules it
        imports."""
        names, modules = {}, {}
        for key, one in self.symbols.items():
            if one.file == path and "." not in one.name:
                names[one.name] = key
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                source = _module_of(set(self.files), path, base)
                for alias in node.names:
                    bound = alias.asname or alias.name
                    # `from library import books`: a module, not a name
                    inner = _module_of(set(self.files), path,
                                       f"{base}.{alias.name}"
                                       if base.strip(".") else
                                       base + alias.name)
                    if inner is not None:
                        modules[bound] = inner
                    elif source is not None and \
                            f"{source}#{alias.name}" in self.symbols:
                        names[bound] = f"{source}#{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found = _module_of(set(self.files), path, alias.name)
                    if found is not None:
                        modules[alias.asname or alias.name] = found
        return names, modules

    def _use(self, path: str, tree) -> None:
        names, modules = self._bound(path, tree)
        # an import is a use, from the module itself
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    key = names.get(alias.asname or alias.name)
                    if key is not None and not key.startswith(f"{path}#"):
                        self.uses.append(Use(key, path, node.lineno, "",
                                             "import"))

        def walk(node, enclosing: str, owner: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk(child, f"{owner}.{child.name}" if owner else
                         child.name)
                    continue
                if isinstance(child, ast.ClassDef):
                    walk(child, child.name, child.name)
                    continue
                self._seen(path, child, enclosing, names, modules, owner)
                walk(child, enclosing, owner)
        walk(tree, "")
        outer = self._outer(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._dispatch(path, node, names)
                self._outside(path, node, outer)
        self._tables(path, tree, names)

    def _seen(self, path: str, node, enclosing: str, names: dict,
              modules: dict, owner: str) -> None:
        how = "read"
        target = None
        parent_call = getattr(node, "_called", False)
        if isinstance(node, ast.Call):
            for one in (node.func,):
                one._called = True
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            target = names.get(node.id)
            if target is None and owner:
                # a method of its own class, by its name alone: not a use
                target = None
        elif isinstance(node, ast.Attribute) and isinstance(
                node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            module = modules.get(node.value.id)
            if module is not None:
                key = f"{module}#{node.attr}"
                target = key if key in self.symbols else None
            elif node.value.id == "self" and owner:
                key = f"{path}#{owner}.{node.attr}"
                target = key if key in self.symbols else None
        if target is None:
            return
        if parent_call:
            how = "call"
        elif isinstance(getattr(node, "_annotation", None), bool):
            how = "type"
        self.uses.append(Use(target, path, node.lineno, enclosing, how))

    def _dispatch(self, path: str, node, names: dict) -> None:
        """The strings a function compares a parameter (or a part of one)
        to: what it answers to."""
        params = {one.arg for one in node.args.args}
        found = []
        for one in ast.walk(node):
            if not isinstance(one, ast.Compare) or len(one.ops) != 1 or \
                    not isinstance(one.ops[0], ast.Eq):
                continue
            left, right = one.left, one.comparators[0]
            for subject, value in ((left, right), (right, left)):
                base = subject
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name) and isinstance(
                        value, ast.Constant) and isinstance(value.value, str):
                    if base.id in params or base.id in self._locals_of(
                            node, params):
                        found.append(value.value)
        if found:
            self.cases.setdefault(f"{path}#{self._name_of(path, node)}",
                                  []).extend(dict.fromkeys(found))

    def _outer(self, tree) -> dict:
        """What a module imports from outside the project: {bound name:
        its module}."""
        out = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    out[alias.asname or alias.name] = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    out[alias.asname or alias.name] = alias.name
        return {name: module for name, module in out.items()
                if _module_of(set(self.files), "", module) is None and
                not module.split(".")[0] == "__future__"}

    def _outside(self, path: str, node, outer: dict) -> None:
        import builtins
        found = []
        for one in ast.walk(node):
            if not isinstance(one, ast.Call):
                continue
            func = one.func
            if isinstance(func, ast.Name) and (
                    func.id in outer or hasattr(builtins, func.id)):
                found.append(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(
                    func.value, ast.Name) and func.value.id in outer:
                found.append(f"{func.value.id}.{func.attr}")
        if found:
            self.outside[f"{path}#{self._name_of(path, node)}"] = list(
                dict.fromkeys(found))

    @staticmethod
    def _locals_of(node, params: set) -> set:
        """Names a function sets from its parameters (`command =
        args[0]`)."""
        out = set()
        for one in ast.walk(node):
            if isinstance(one, ast.Assign) and any(
                    isinstance(name, ast.Name) and name.id in params
                    for name in ast.walk(one.value)):
                out |= {target.id for target in one.targets
                        if isinstance(target, ast.Name)}
        return out

    def _name_of(self, path: str, node) -> str:
        for key, one in self.symbols.items():
            if one.file == path and one.start == node.lineno and \
                    one.kind in ("function", "method"):
                return one.name
        return node.name

    def _tables(self, path: str, tree, names: dict) -> None:
        """A module's table of functions by string (`ROUTES = {"/plants":
        _plants}`): its keys are what the functions that look it up answer
        to."""
        tables = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(
                    node.value, ast.Dict) and node.value.keys and all(
                    isinstance(key, ast.Constant) and
                    isinstance(key.value, str) for key in node.value.keys):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        tables[target.id] = [key.value
                                             for key in node.value.keys]
        if not tables:
            return
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            used = {one.id for one in ast.walk(node)
                    if isinstance(one, ast.Name) and one.id in tables}
            for table in sorted(used):
                self.cases.setdefault(f"{path}#{self._name_of(path, node)}",
                                      []).extend(tables[table])

    # -- looked up ----------------------------------------------------------------
    def find(self, name: str) -> list:
        """The symbols called `name` -- or `Class.name` -- [Symbol]."""
        return [one for one in self.symbols.values()
                if one.name == name or one.name.endswith("." + name)]

    def used_by(self, name: str) -> list:
        """Each use of what is called `name`: [Use]."""
        keys = {f"{one.file}#{one.name}" for one in self.find(name)}
        return [one for one in self.uses if one.symbol in keys]

    def fields(self, name: str) -> list:
        return [one for one in self.symbols.values()
                if one.kind == "field" and one.owner == name]

    def methods(self, name: str) -> list:
        return [one for one in self.symbols.values()
                if one.kind == "method" and one.owner == name and
                not one.name.split(".")[-1].startswith("__")]

    def answers_to(self, name: str) -> list:
        keys = {f"{one.file}#{one.name}" for one in self.find(name)}
        return [case for key in sorted(keys)
                for case in self.cases.get(key, ())]

    @staticmethod
    def is_test(path: str) -> bool:
        base = path.rsplit("/", 1)[-1]
        return base.startswith("test_") or base.endswith("_test.py") or \
            "/tests/" in f"/{path}" or path.startswith("tests/")


def renamed(files: dict, index: Index, old: str, new: str) -> dict:
    """Every Python file that defines or uses `old` (a function, class or
    constant of the project), its name `old` made `new` -- as code, never
    in strings or comments (`tokenize`): {path: changed text}. Empty where
    the project has no such name, or already has `new`."""
    import io
    import tokenize
    defined = [one for one in index.find(old) if "." not in one.name]
    if not defined or index.find(new):
        return {}
    keys = {f"{one.file}#{one.name}" for one in defined}
    paths = {one.file for one in defined} | {
        use.file for use in index.uses if use.symbol in keys}
    out = {}
    for path in sorted(paths):
        text = files[path]
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return {}
        lines = text.splitlines(keepends=True)
        # from the end, so that what is replaced keeps the columns before
        for token in reversed(tokens):
            if token.type == tokenize.NAME and token.string == old:
                row, column = token.start
                line = lines[row - 1]
                lines[row - 1] = line[:column] + new + \
                    line[column + len(old):]
        changed = "".join(lines)
        if changed != text:
            out[path] = changed
    return out
