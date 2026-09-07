"""Connectives and quantifiers, evaluated in three values.

V686 answered a polar question by taking the first property that matched and
returning. That is right for one property and wrong for two: `does a dog have
a tail and wings` came back VERIFIED on the strength of the tail. The fix is
not a loop -- it is admitting that a question has *structure*, and that the
structure needs a logic to evaluate it in.

Two values will not do. This system has three real answers -- the norms state
it, the norms deny it, the norms are silent -- and silence is not falsehood.
So the connectives are Kleene's strong three-valued ones:

    and     FALSE if any conjunct is false, whatever the others say
            UNKNOWN if none is false but one is unknown
            TRUE only if every one is true

    or      TRUE if any disjunct is true, whatever the others say
            UNKNOWN if none is true but one is unknown
            FALSE only if every one is false

    not     swaps true and false; unknown stays unknown

The asymmetry is the point. `is a dog furry or purple` is TRUE even though
nothing is recorded about purple dogs, because one true disjunct settles it.
`does a dog have a tail and wings` is FALSE because one false conjunct settles
it. And `is a dog furry and telepathic` is UNKNOWN, not false: the system does
not know, and saying so is the whole of R8.

Quantifiers work over the kinds beneath a concept, which is where v687 gets
`do all birds fly` right where a single lookup got it wrong. `all` fails on
one counterexample, `some` succeeds on one witness, `no` is `not some`, and
`most` is a count rather than a claim -- reported as a proportion, because
"most birds fly, and here are the two that do not" is a better answer than
either yes or no.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"

#: Words that turn a claim into its denial.
NEGATORS = frozenset("""
not cannot cant never no without lacks lacking non isnt arent doesnt dont
""".split())

#: Quantifiers, and how many of the kinds beneath a concept each demands.
QUANTIFIERS = {
    "all": "all", "every": "all", "each": "all", "any": "all", "always": "all",
    "some": "some", "certain": "some", "sometimes": "some",
    "no": "none", "none": "none", "never": "none",
    "most": "most", "usually": "most", "typically": "most", "generally": "most",
}


@dataclass
class Node:
    """One node of the query: a term, or a connective over other nodes."""
    op: str                                   # term | and | or | not
    term: str = ""
    children: list["Node"] = field(default_factory=list)
    #: Filled in by `evaluate`, so the page can show which half failed.
    value: str = UNKNOWN
    detail: str = ""

    def terms(self) -> list[str]:
        if self.op == "term":
            return [self.term]
        return [term for child in self.children for term in child.terms()]

    def as_dict(self) -> dict[str, Any]:
        return {"op": self.op, "term": self.term, "value": self.value,
                "detail": self.detail,
                "children": [child.as_dict() for child in self.children]}


@dataclass
class Query:
    """A parsed question: how much of the class it claims, and of what."""
    quantifier: str | None            # all | some | none | most, or None
    tree: Node
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"quantifier": self.quantifier, "tree": self.tree.as_dict(),
                "text": self.text}


def negate(value: str) -> str:
    if value == TRUE:
        return FALSE
    if value == FALSE:
        return TRUE
    return UNKNOWN


def conjoin(values: list[str]) -> str:
    """Kleene AND: one false settles it, one unknown suspends it."""
    if FALSE in values:
        return FALSE
    if UNKNOWN in values:
        return UNKNOWN
    return TRUE if values else UNKNOWN


def disjoin(values: list[str]) -> str:
    """Kleene OR: one true settles it, one unknown suspends it."""
    if TRUE in values:
        return TRUE
    if UNKNOWN in values:
        return UNKNOWN
    return FALSE if values else UNKNOWN


def evaluate(node: Node, test: Callable[[str], tuple[str, str]]) -> str:
    """Walk the tree, calling `test` on each term. Values are kept on the
    nodes so the page can show which half of an `and` actually failed."""
    if node.op == "term":
        node.value, node.detail = test(node.term)
        return node.value
    values = [evaluate(child, test) for child in node.children]
    if node.op == "not":
        node.value = negate(values[0] if values else UNKNOWN)
    elif node.op == "or":
        node.value = disjoin(values)
    else:
        node.value = conjoin(values)
    return node.value


# -- reading a question into a tree ---------------------------------------
def _atom(phrase: str, aside: frozenset[str]) -> Node | None:
    """One side of a connective: its content words, negated if it says so.

    A phrase is one claim even when it takes several words -- `a red breast`
    is not two properties -- so the words are kept together and matching is
    left to the caller, which knows how a property is stored.
    """
    words = [word for word in re.findall(r"[a-z]+", phrase.lower())]
    denied = any(word in NEGATORS for word in words)
    content = [word for word in words
               if word not in aside and word not in NEGATORS
               and word not in QUANTIFIERS]
    if not content:
        return None
    claim = Node(op="and", children=[Node(op="term", term=word)
                                     for word in content]) \
        if len(content) > 1 else Node(op="term", term=content[0])
    return Node(op="not", children=[claim]) if denied else claim


def parse(text: str, aside: frozenset[str]) -> Query:
    """Read `a tail and wings`, `furry or hairless`, `not furry`.

    Splitting on the words rather than parsing a grammar is deliberate: the
    questions this answers are short, and a real parse of coordination is a
    research project that would not change any answer here.
    """
    lowered = " " + text.lower().strip().rstrip("?") + " "
    quantifier = None
    for word, kind in QUANTIFIERS.items():
        if re.search(rf"\b{word}\b", lowered):
            quantifier = kind
            # `no birds fly` is quantified, not negated. Leaving the word in
            # would negate the claim as well, answering the opposite question.
            lowered = re.sub(rf"\b{word}\b", " ", lowered)
            break

    groups = re.split(r"\s+or\s+|\s*,\s*or\s+", lowered)
    disjuncts: list[Node] = []
    for group in groups:
        parts = re.split(r"\s+and\s+|\s*,\s+", group)
        atoms = [node for node in (_atom(part, aside) for part in parts)
                 if node is not None]
        if not atoms:
            continue
        disjuncts.append(atoms[0] if len(atoms) == 1
                         else Node(op="and", children=atoms))
    if not disjuncts:
        tree = Node(op="and", children=[])
    elif len(disjuncts) == 1:
        tree = disjuncts[0]
    else:
        tree = Node(op="or", children=disjuncts)
    return Query(quantifier=quantifier, tree=tree, text=text.strip())


#: R18. Constructions no rule here covers. Each is refused by name rather
#: than answered from the part of the question that happens to be
#: understandable, which is how `is a whale bigger than a dolphin` came back
#: VERIFIED -- the comparative was dropped and the leftover words were looked
#: up. Every silent wrong answer in the v686 audit had this shape.
UNSUPPORTED: tuple[tuple[str, str], ...] = (
    (r"\b(bigger|smaller|larger|heavier|lighter|faster|slower|older|younger|"
     r"stronger|taller|shorter|better|worse|more|less)\s+than\b",
     "a comparative. Nothing in this data has a magnitude: AwA2 records "
     "`big` as a yes or no and XCSLB records `is the largest animal` as a "
     "string. There is no scale to compare on"),
    (r"\bhow\s+many\b(?!\s+kinds?\b)",
     "a count of parts or instances. The ontology holds no numbers -- it can "
     "count *kinds* over the taxonomy, so `how many kinds of dog` is "
     "answerable, but `how many legs` is not stored anywhere"),
    (r"\bthe\s+(biggest|largest|smallest|fastest|oldest|best|most)\b",
     "a superlative, which needs an ordering over concepts. The same missing "
     "scale as a comparative"),
    (r"\bwhat\s+if\b|\bwould\s+have\b|\bhad\s+been\b",
     "a counterfactual. Every rule here reasons about what is recorded, and "
     "nothing supports reasoning about what is not"),
    (r"^\s*when\b|\bwhat\s+year\b|\bwhat\s+date\b|\bhow\s+long\s+ago\b",
     "a date. Nothing here is in time: the ontology holds concepts and the "
     "relations between them, and not one fact in it carries a when"),
    (r"\bwhat\s+do(?:es)?\b[^?]*\bmean\b|\bhow\s+do\s+you\s+"
     r"(?:spell|pronounce|say|write)\b|\bhow\s+many\s+letters\b",
     "a question about the word rather than the thing. This reasons over "
     "senses, which are what words point at; it holds nothing about the "
     "words themselves"),
    (r"\bwho\s+(?:invented|discovered|wrote|founded|created|built|painted|"
     r"composed|designed|won|said)\b|\bthe\s+capital\s+of\b|"
     r"\bwhat\s+is\s+the\s+(?:capital|population|address|name)\s+of\b",
     "a fact about a named individual. The taxonomy holds kinds, not "
     "instances -- there is no Alexander Bell in it, and asking who invented "
     "the telephone can only be answered by whichever concept the crawl "
     "happened to hang the phrase on"),
    (r"\bthe\s+opposite\s+of\b|\bantonyms?\s+of\b|\bopposite\s+(?:to|from)\b",
     "an antonym. WordNet records antonymy between word forms, and this "
     "store keeps only relations between senses, so the opposite of a thing "
     "is not something it can look up"),
)


def unsupported(question: str) -> str | None:
    """R18: name the construction this cannot answer, or None.

    Refusing is the feature. A system that answers an easier question than
    the one asked, without saying so, is worse than one that says no -- and
    R8 already commits to VERIFIED, CONTRADICTED or UNKNOWN with nothing in
    between.
    """
    text = (question or "").strip().lower().rstrip("?")
    for pattern, reason in UNSUPPORTED:
        if re.search(pattern, text):
            return reason
    return None


def readable(node: Node) -> str:
    """The tree back as words, for the note under the answer."""
    if node.op == "term":
        return f"“{node.term}”"
    if node.op == "not":
        return "not " + readable(node.children[0])
    joiner = " or " if node.op == "or" else " and "
    inner = joiner.join(readable(child) for child in node.children)
    return f"({inner})" if len(node.children) > 1 else inner
