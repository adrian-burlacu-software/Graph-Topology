"""Python's types and the engine's: one vocabulary inside, each language's
words at its edge.

The engine types a tree as TypeScript writes types (`number`, `string`,
`boolean`, `T[]`, `[A, B]`, `Set<T>`, `Map<K, V>`, `T | undefined`): its
library, its forms and its search are keyed by them. Python's annotations
are read into them (`engine`), and written back from them where a Python
program is printed (`written`) -- with how the person wrote a type kept
where they wrote one (`int` is not `float`, though both are a number).

    engine("list[int]")            -> "number[]"
    engine("tuple[str, int]")      -> "[string, number]"
    written("number[]")            -> "list[int]"
    written("number", "float")     -> "float"
"""
from __future__ import annotations

import ast

SCALARS = {"int": "number", "float": "number", "complex": "number",
           "str": "string", "bool": "boolean", "None": "undefined",
           "NoneType": "undefined", "Any": "any", "object": "any"}
BACK = {"number": "int", "string": "str", "boolean": "bool",
        "undefined": "None", "any": "Any", "void": "None"}


def engine(annotation: str | None) -> str:
    """A Python annotation, as the engine types it; `any` where it says
    nothing the engine has a word for."""
    if not annotation:
        return "any"
    try:
        node = ast.parse(annotation.strip(), mode="eval").body
    except SyntaxError:
        return "any"
    return _engine(node)


def _name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _engine(node) -> str:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _engine(node.left), _engine(node.right)
        if right == "undefined":
            return f"{left} | undefined"
        if left == "undefined":
            return f"{right} | undefined"
        return f"{left} | {right}"
    if isinstance(node, ast.Subscript):
        outer = _name(node.value)
        inner = node.slice
        parts = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
        if outer in ("list", "List", "Sequence", "Iterable", "Iterator",
                     "Generator", "deque", "MutableSequence"):
            return _array(_engine(parts[0]))
        if outer in ("tuple", "Tuple"):
            if len(parts) == 2 and isinstance(parts[1], ast.Constant) and \
                    parts[1].value is Ellipsis:
                return _array(_engine(parts[0]))
            return "[" + ", ".join(_engine(one) for one in parts) + "]"
        if outer in ("set", "Set", "frozenset", "FrozenSet", "AbstractSet"):
            return f"Set<{_engine(parts[0])}>"
        if outer in ("dict", "Dict", "Mapping", "MutableMapping",
                     "defaultdict", "OrderedDict", "Counter"):
            if outer == "Counter":
                return f"Map<{_engine(parts[0])}, number>"
            return (f"Map<{_engine(parts[0])}, "
                    f"{_engine(parts[1]) if len(parts) > 1 else 'any'}>")
        if outer == "Optional":
            return f"{_engine(parts[0])} | undefined"
        if outer == "Union":
            found = [_engine(one) for one in parts]
            rest = [one for one in found if one != "undefined"]
            out = " | ".join(rest) or "any"
            return out + (" | undefined" if "undefined" in found else "")
        if outer == "Callable":
            return "fn:" + (_engine(parts[1]) if len(parts) > 1 else "any")
        return "any"
    name = _name(node)
    if name in SCALARS:
        return SCALARS[name]
    if name in ("list", "List"):
        return "any[]"
    if name in ("tuple", "Tuple"):
        return "any[]"
    if name in ("dict", "Dict"):
        return "Map<any, any>"
    if name in ("set", "Set"):
        return "Set<any>"
    return "any"


def _array(inner: str) -> str:
    return f"({inner})[]" if "|" in inner else f"{inner}[]"


def written(kind: str, hint: str | None = None) -> str:
    """An engine type as Python writes it -- as the person wrote it, where
    they did (`hint`)."""
    if hint:
        return hint
    kind = (kind or "any").strip()
    if kind.endswith(" | undefined"):
        return f"{written(kind[:-len(' | undefined')])} | None"
    if kind in BACK:
        return BACK[kind]
    if "|" in kind and not kind.endswith(("[]", "]", ">")):
        return " | ".join(written(one.strip()) for one in kind.split("|"))
    if kind.endswith("[]"):
        inner = kind[:-2]
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        return f"list[{written(inner)}]"
    if kind.startswith("[") and kind.endswith("]"):
        return "tuple[" + ", ".join(written(one) for one in
                                    _split(kind[1:-1])) + "]"
    for outer, said in (("Set<", "set"), ("Map<", "dict")):
        if kind.startswith(outer) and kind.endswith(">"):
            inner = _split(kind[len(outer):-1])
            return f"{said}[{', '.join(written(one) for one in inner)}]"
    if kind.startswith("fn:"):
        return "Callable[..., " + written(kind[3:]) + "]"
    return "Any"


def _split(text: str) -> list:
    """Top-level comma-separated parts of a type list."""
    out, depth, start = [], 0, 0
    for at, char in enumerate(text):
        if char in "[<(":
            depth += 1
        elif char in "]>)":
            depth -= 1
        elif char == "," and depth == 0:
            out.append(text[start:at].strip())
            start = at + 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return out
