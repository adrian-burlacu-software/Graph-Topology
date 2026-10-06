"""Python's library, as the search composes it: operators over the engine's
types, each with how Python writes it (`Op.py`).

TypeScript's is read from its compiler's declarations (`program.library`);
Python's is chosen here, member by member -- the builtins, the `str` and
`list` methods, `math`, the operators -- each a typed action the search can
take, written as people write Python. Where a member does what one of
TypeScript's does, it has that one's key (`method:join:string[],string->
string`), so what the search knows of composing it holds in both languages;
how it is written is Python's (`sep.join(xs)`).

Forms (a member with a callback) are the comprehension and its kin:
`map`, `filter`, `reduce`, `some`/`every` (any, all), `find`, `findIndex`,
and Python's own with a key -- `sortedBy`, `sumOf`, `maxBy`, `minBy`
(`pyprint.py` writes them).
"""
from __future__ import annotations

from research.v696 import program as P

SCALARS = ("number", "string", "boolean")


def _op(name, kind, needs, gives, py) -> P.Op:
    return P.Op(name, kind, tuple(needs), gives, py=py)


def _members(types) -> list:
    """Every member, as (name, kind, needs, gives, template)."""
    arrays = [one for one in types if one.endswith("[]")]
    out = [
        # -- operators -------------------------------------------------------
        ("+", "operator", ("number", "number"), "number", "({0} + {1})"),
        ("-", "operator", ("number", "number"), "number", "({0} - {1})"),
        ("*", "operator", ("number", "number"), "number", "({0} * {1})"),
        ("/", "operator", ("number", "number"), "number", "({0} / {1})"),
        ("//", "operator", ("number", "number"), "number", "({0} // {1})"),
        ("%", "operator", ("number", "number"), "number", "({0} % {1})"),
        ("**", "operator", ("number", "number"), "number", "({0} ** {1})"),
        ("-", "operator", ("number",), "number", "(-{0})"),
        ("+", "operator", ("string", "string"), "string", "({0} + {1})"),
        ("*", "operator", ("string", "number"), "string", "({0} * {1})"),
        ("===", "operator", ("number", "number"), "boolean", "({0} == {1})"),
        ("===", "operator", ("string", "string"), "boolean", "({0} == {1})"),
        ("===", "operator", ("boolean", "boolean"), "boolean",
         "({0} == {1})"),
        ("!==", "operator", ("number", "number"), "boolean", "({0} != {1})"),
        ("!==", "operator", ("string", "string"), "boolean", "({0} != {1})"),
        ("<", "operator", ("number", "number"), "boolean", "({0} < {1})"),
        (">", "operator", ("number", "number"), "boolean", "({0} > {1})"),
        ("<=", "operator", ("number", "number"), "boolean", "({0} <= {1})"),
        (">=", "operator", ("number", "number"), "boolean", "({0} >= {1})"),
        ("<", "operator", ("string", "string"), "boolean", "({0} < {1})"),
        (">", "operator", ("string", "string"), "boolean", "({0} > {1})"),
        ("<=", "operator", ("string", "string"), "boolean",
         "({0} <= {1})"),
        (">=", "operator", ("string", "string"), "boolean",
         "({0} >= {1})"),
        ("^", "operator", ("number", "number"), "number", "({0} ^ {1})"),
        ("&", "operator", ("number", "number"), "number", "({0} & {1})"),
        ("|", "operator", ("number", "number"), "number", "({0} | {1})"),
        ("<<", "operator", ("number", "number"), "number",
         "({0} << {1})"),
        (">>", "operator", ("number", "number"), "number",
         "({0} >> {1})"),
        ("~", "operator", ("number",), "number", "(~{0})"),
        ("math.pow", "function", ("number", "number"), "number",
         "math.pow({0}, {1})"),
        ("divmod quotient", "function", ("number", "number"), "number",
         "divmod({0}, {1})[0]"),
        ("&&", "operator", ("boolean", "boolean"), "boolean",
         "({0} and {1})"),
        ("||", "operator", ("boolean", "boolean"), "boolean",
         "({0} or {1})"),
        ("!", "operator", ("boolean",), "boolean", "(not {0})"),
        ("in", "operator", ("string", "string"), "boolean", "({0} in {1})"),
        # -- numbers ---------------------------------------------------------
        ("abs", "function", ("number",), "number", "abs({0})"),
        ("round", "function", ("number",), "number", "round({0})"),
        ("round", "function", ("number", "number"), "number",
         "round({0}, {1})"),
        ("min", "function", ("number", "number"), "number", "min({0}, {1})"),
        ("max", "function", ("number", "number"), "number", "max({0}, {1})"),
        ("pow", "function", ("number", "number"), "number", "pow({0}, {1})"),
        ("Math.floor", "function", ("number",), "number",
         "math.floor({0})"),
        ("Math.ceil", "function", ("number",), "number", "math.ceil({0})"),
        ("Math.sqrt", "function", ("number",), "number", "math.sqrt({0})"),
        ("math.isqrt", "function", ("number",), "number",
         "math.isqrt({0})"),
        ("math.gcd", "function", ("number", "number"), "number",
         "math.gcd({0}, {1})"),
        ("math.lcm", "function", ("number", "number"), "number",
         "math.lcm({0}, {1})"),
        ("math.factorial", "function", ("number",), "number",
         "math.factorial({0})"),
        ("math.comb", "function", ("number", "number"), "number",
         "math.comb({0}, {1})"),
        ("Math.log", "function", ("number",), "number", "math.log({0})"),
        ("math.log2", "function", ("number",), "number", "math.log2({0})"),
        ("math.log10", "function", ("number",), "number",
         "math.log10({0})"),
        ("int", "function", ("number",), "number", "int({0})"),
        ("float", "function", ("number",), "number", "float({0})"),
        ("bin", "function", ("number",), "string", "bin({0})[2:]"),
        ("hex", "function", ("number",), "string", "hex({0})[2:]"),
        ("divides", "operator", ("number", "number"), "boolean",
         "({0} % {1} == 0)"),
        # -- conversions -----------------------------------------------------
        ("String", "function", ("number",), "string", "str({0})"),
        ("parseInt", "function", ("string",), "number", "int({0})"),
        ("parseFloat", "function", ("string",), "number", "float({0})"),
        ("ord", "function", ("string",), "number", "ord({0})"),
        ("chr", "function", ("number",), "string", "chr({0})"),
        ("Boolean", "function", ("number",), "boolean", "bool({0})"),
        ("Boolean", "function", ("string",), "boolean", "bool({0})"),
        # -- strings ---------------------------------------------------------
        ("length", "property", ("string",), "number", "len({0})"),
        ("toUpperCase", "method", ("string",), "string", "{0}.upper()"),
        ("toLowerCase", "method", ("string",), "string", "{0}.lower()"),
        ("trim", "method", ("string",), "string", "{0}.strip()"),
        ("trimStart", "method", ("string",), "string", "{0}.lstrip()"),
        ("trimEnd", "method", ("string",), "string", "{0}.rstrip()"),
        ("title", "method", ("string",), "string", "{0}.title()"),
        ("capitalize", "method", ("string",), "string", "{0}.capitalize()"),
        ("swapcase", "method", ("string",), "string", "{0}.swapcase()"),
        ("split", "method", ("string",), "string[]", "{0}.split()"),
        ("split", "method", ("string", "string"), "string[]",
         "{0}.split({1})"),
        ("replaceAll", "method", ("string", "string", "string"), "string",
         "{0}.replace({1}, {2})"),
        ("startsWith", "method", ("string", "string"), "boolean",
         "{0}.startswith({1})"),
        ("endsWith", "method", ("string", "string"), "boolean",
         "{0}.endswith({1})"),
        ("indexOf", "method", ("string", "string"), "number",
         "{0}.find({1})"),
        ("count", "method", ("string", "string"), "number",
         "{0}.count({1})"),
        ("isdigit", "method", ("string",), "boolean", "{0}.isdigit()"),
        ("isalpha", "method", ("string",), "boolean", "{0}.isalpha()"),
        ("isalnum", "method", ("string",), "boolean", "{0}.isalnum()"),
        ("isupper", "method", ("string",), "boolean", "{0}.isupper()"),
        ("islower", "method", ("string",), "boolean", "{0}.islower()"),
        ("isspace", "method", ("string",), "boolean", "{0}.isspace()"),
        ("zfill", "method", ("string", "number"), "string",
         "{0}.zfill({1})"),
        ("repeat", "method", ("string", "number"), "string",
         "({0} * {1})"),
        ("reversed", "method", ("string",), "string", "{0}[::-1]"),
        ("slice", "method", ("string", "number"), "string", "{0}[{1}:]"),
        ("slice", "method", ("string", "number", "number"), "string",
         "{0}[{1}:{2}]"),
        ("prefix", "method", ("string", "number"), "string", "{0}[:{1}]"),
        ("every other", "method", ("string",), "string", "{0}[::2]"),
        ("chars", "function", ("string",), "string[]", "list({0})"),
        ("join", "method", ("string[]", "string"), "string",
         "{1}.join({0})"),
        ("concat", "function", ("string[]",), "string", "''.join({0})"),
        ("sorted", "function", ("string",), "string",
         "''.join(sorted({0}))"),
        ("unique", "function", ("string",), "string",
         "''.join(dict.fromkeys({0}))"),
    ]
    for kind in arrays:
        item = kind[:-2]
        out += [
            ("length", "property", (kind,), "number", "len({0})"),
            ("slice", "method", (kind, "number"), kind, "{0}[{1}:]"),
            ("slice", "method", (kind, "number", "number"), kind,
             "{0}[{1}:{2}]"),
            ("prefix", "method", (kind, "number"), kind, "{0}[:{1}]"),
            ("reverse", "method", (kind,), kind, "{0}[::-1]"),
            ("sorted", "function", (kind,), kind, "sorted({0})"),
            ("sorted descending", "function", (kind,), kind,
             "sorted({0}, reverse=True)"),
            ("unique", "function", (kind,), kind,
             "list(dict.fromkeys({0}))"),
            ("distinct count", "function", (kind,), "number",
             "len(set({0}))"),
            ("concat", "method", (kind, kind), kind, "({0} + {1})"),
            ("includes", "method", (kind, item), "boolean", "({1} in {0})"),
            ("count", "method", (kind, item), "number", "{0}.count({1})"),
            ("indexOf", "method", (kind, item), "number", "{0}.index({1})"),
            ("every other", "method", (kind,), kind, "{0}[::2]"),
            ("repeat", "method", (kind, "number"), kind, "({0} * {1})"),
            ("===", "operator", (kind, kind), "boolean", "({0} == {1})"),
            ("!==", "operator", (kind, kind), "boolean", "({0} != {1})"),
        ]
        if item == "number":
            out += [("sum", "function", (kind,), "number", "sum({0})"),
                    ("Math.max", "function", (kind,), "number", "max({0})"),
                    ("Math.min", "function", (kind,), "number", "min({0})"),
                    ("product", "function", (kind,), "number",
                     "math.prod({0})"),
                    ("mean", "function", (kind,), "number",
                     "statistics.mean({0})")]
        if item == "string":
            out += [("Math.max", "function", (kind,), "string", "max({0})"),
                    ("Math.min", "function", (kind,), "string", "min({0})")]
        if item == "boolean":
            out += [("all", "function", (kind,), "boolean", "all({0})"),
                    ("any", "function", (kind,), "boolean", "any({0})")]
        if f"{item}[]" == kind and "number" in types:
            out += [("of", "function", (item,), kind, "[{0}]")]
    return out


#: forms: (name, scope after the element, body, gives) per array type;
#: "U" is free, "T" the element's type
FORMS = (
    ("map", (("x", "T"), ("i", "number")), "U", "U[]"),
    ("filter", (("x", "T"), ("i", "number")), "boolean", "T[]"),
    ("some", (("x", "T"), ("i", "number")), "boolean", "boolean"),
    ("every", (("x", "T"), ("i", "number")), "boolean", "boolean"),
    ("find", (("x", "T"), ("i", "number")), "boolean", "T"),
    ("findIndex", (("x", "T"), ("i", "number")), "boolean", "number"),
    ("sumOf", (("x", "T"), ("i", "number")), "number", "number"),
    ("sortedBy", (("x", "T"),), "number", "T[]"),
    ("maxBy", (("x", "T"),), "number", "T"),
    ("minBy", (("x", "T"),), "number", "T"),
)


_LIBRARY: dict = {}


def library(types=P.TYPES) -> P.Library:
    key = tuple(sorted(types))
    if key in _LIBRARY:
        return _LIBRARY[key]
    wanted = set(types)
    ops, seen = [], set()
    for name, kind, needs, gives, py in _members(types):
        op = _op(name, kind, needs, gives, py)
        if op.key not in seen and op.needs and all(
                one in wanted for one in op.needs) and op.gives in wanted:
            seen.add(op.key)
            ops.append(op)
    # the engine's own: the counting loop's numbers, an element, one more
    # at the end, `a if c else b`
    if "number" in wanted and "number[]" in wanted:
        ops.append(P.RANGE)
    for op in P.INDEX + P.APPEND:
        if all(one in wanted for one in op.needs) and op.gives in wanted:
            ops.append(op)
    for kind in types:
        ops.append(P.Op("?:", "ternary", ("boolean", kind, kind), kind))
    forms = []
    for kind in types:
        if not kind.endswith("[]"):
            continue
        item = kind[:-2]
        for name, scope, body, gives in FORMS:
            scope = tuple((one, item if what == "T" else what)
                          for one, what in scope)
            forms.append(P.Form(name, kind, scope,
                                item if body == "T" else body, (),
                                gives.replace("T", item)))
        for start in (False, True):
            # reduce: (acc, x, i) from its first, or from a value given
            forms.append(P.Form("reduce", kind, (("acc", item), ("x", item),
                                                 ("i", "number")),
                                item, (item,) if start else (), item))
        forms.append(P.Form("reduce", kind, (("acc", "U"), ("x", item),
                                             ("i", "number")),
                            "U", ("U",), "U"))
    found = P.Library(ops, forms, [])
    _LIBRARY[key] = found
    return found
