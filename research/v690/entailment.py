"""EntailmentBank: build the proof tree with the executive (F3).

`DESIGN.md` §8c's F3 asked whether the depth question is settled by a task
rather than a corpus. ProofWriter answered the first half -- the executive
searches to depth 5 without a miss when the rules are given -- and this is
the second half, where they are not.

EntailmentBank gives a hypothesis and the sentences a proof of it needs, and
the gold tree says which of them combine at each step. Nothing says *why*
they combine: unlike ProofWriter there are no rules, and a step's conclusion
is new prose (`earth is a planet that rotates on its tilted axis`). Writing
that prose is a generation problem and is not attempted here. What is built,
and scored, is the **structure**: which sentences group together on the way
to the hypothesis.

## The operators

Each way two nodes can belong together is an operator, and they compete for
the same merge -- which is the thing `AUDIT.md`'s credit work kept wanting
and bAbI never supplied, since there the act decides the operator and
nothing else proposes. Here four of them propose on most cycles and the
executive picks by utility:

    ground      a taxonomic sentence and what it is about. `earth is a kind
                of planet` with `the earth rotates on its tilted axis`.
    elaborate   two sentences about the same thing, which is one step:
                `the new moon is when ...` twice over.
    join        the plainest reason -- the two nodes sharing most words.
    apply       what is left, once nothing more specific proposes: the
                general rule is the last thing applied, and `chain` gets
                that backwards.

The order was read off the gold trees and then measured. See `SIGNALS`.

## What is measured

A step is named by the set of original sentences beneath it, so the score
does not depend on what the intermediates are called:

    gold   sent1 & sent2 -> int1;  int1 & sent3 -> hypothesis
    steps  {sent1,sent2}  and  {sent1,sent2,sent3}

**The root step is free** in task 1, where every sentence given is needed
and the root is therefore always all of them: putting everything into one
step scores 47.9% steps F1 at 100% precision and recovers no structure at
all. `inner` drops it, and the inner numbers are the ones that mean
anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from research.v687.executive import (ANSWERED, CONTINUE, DECLINED, Executive,
                                     Operator, Working)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "entailmentbank" / "entailment_trees_emnlp2021_data_v3"
DATASET = DATA / "dataset"

#: Function words, dropped before anything is compared. Short and fixed: a
#: stemmer here would make `rotates`/`rotation` one word and the gold trees
#: turn on exactly that distinction.
STOP = frozenset("""a an the of to in on is are was were be been being and
or as at by for from into its it that this these those with will would can
could may might not no s than then there their them they he she his her
have has had do does did what which who whom when where why how if else
one two some any all each every other another such very more most less
least much many own same so only just also both few""".split())

#: How a sentence says it is taxonomy. `is a X located in Y` is here because
#: `new york is a state located in the united states` grounds new york the
#: same way `earth is a kind of planet` grounds earth.
TAXONOMIC = re.compile(
    r"\bis a kind of\b|\bis an? example of\b|\bis a type of\b|"
    r"\bis a form of\b|\bare a kind of\b|\bis a\b.*\blocated in\b")

#: The same markers, as a splitter, to get the two sides apart.
SPLIT = re.compile(
    r"\bis a kind of\b|\bis an? example of\b|\bis a type of\b|"
    r"\bis a form of\b|\bare a kind of\b")

#: Measured on task 1 dev, 187 trees, against the two baselines in
#: `baselines` -- one step for everything (inner F1 **0.0%**, it recovers
#: nothing) and left-to-right chaining (**10.6%**):
#:
#:     ground + elaborate + join + apply    inner F1 19.9%   whole tree 34.2%
#:
#: Ablated one at a time on the same set, and two signals that looked
#: obvious from reading four gold trees are **not** in the list because
#: they were measured and cost points: penalising a taxonomic sentence for
#: sharing its *class* with the other node (-0.6 inner F1), and merging two
#: sentences with the same opening words (-1.3). `ground`'s bonus for the
#: taxonomic sentence's *subject* appearing in the other node is worth
#: +1.1 and is kept.
#:
#: **The ceiling is 72%**, not 100%: scoring merges by agreement with the
#: gold tree, binary merges rebuild it exactly in 135 of 187. The rest need
#: a step of three or more. So 38 points sit behind a better reason to
#: merge and none behind more search -- a beam of width 8 over this same
#: affinity scores **worse** than taking the best merge each time (18.4%),
#: because it finds better-scoring wrong trees. The bottleneck is knowing
#: whether two facts compose, which is a judgement about meaning and not
#: about control.
SIGNALS = {"taxonomic-pairs with non-taxonomic": 1.5,
           "both taxonomic": 0.5,
           "the taxonomy's subject is in the other": 2.5,
           "generality, held back": -1.2}


def words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def taxonomy(text: str) -> tuple:
    """(the thing, the class) of `X is a kind of Y`, else (None, None)."""
    parts = SPLIT.split(text.lower(), maxsplit=1)
    if len(parts) != 2:
        return None, None
    return words(parts[0]), words(parts[1])


def generality(text: str, hypothesis: set) -> float:
    """How much this reads as a rule about a class rather than as a fact
    about the thing asked: a general rule shares its predicate with the
    hypothesis and not its subject."""
    mine = words(text)
    if not mine:
        return 1.0
    return 1.0 - len(mine & hypothesis) / len(mine)


@dataclass
class Node:
    """One sentence, or one conclusion drawn from several."""

    name: str
    leaves: frozenset
    words: set
    taxo: bool = False
    thing: set | None = None
    general: float = 1.0


# -- the proof, as a string and as a set of steps --------------------------

STEP = re.compile(r"([^;]+?)\s*->\s*([^;]+)")


def steps_of(proof: str) -> list:
    """[(premise ids, conclusion id)] of `sent1 & sent2 -> int1; ...`."""
    found = []
    for body, head in STEP.findall(proof):
        premises = [one.strip() for one in body.split("&") if one.strip()]
        head = head.strip().split(":")[0].strip()
        if premises and head:
            found.append((premises, head))
    return found


def signature(proof: str) -> set:
    """The tree as a set of steps, each named by the sentences beneath it."""
    steps = steps_of(proof)
    beneath: dict[str, set] = {}

    def expand(node, seen=()):
        if node.startswith("sent"):
            return {node}
        if node in beneath:
            return beneath[node]
        if node in seen:
            return set()
        for premises, head in steps:
            if head == node:
                got: set = set()
                for one in premises:
                    got |= expand(one, tuple(seen) + (node,))
                beneath[node] = got
                return got
        return set()

    for _, head in steps:
        expand(head)
    out = set()
    for premises, head in steps:
        got = set()
        for one in premises:
            got |= ({one} if one.startswith("sent")
                    else beneath.get(one, set()))
        if got:
            out.add(frozenset(got))
    return out


def inner(tree: set) -> set:
    """The tree without its root step, which task 1 gives away for nothing."""
    if not tree:
        return tree
    root = max(tree, key=len)
    return {one for one in tree if one is not root}


# -- the executive ---------------------------------------------------------

def affinity(left: Node, right: Node) -> float:
    """How much these two belong in one step."""
    shared = len(left.words & right.words)
    if not shared:
        return -1.0
    score = float(shared)
    if left.taxo != right.taxo:
        score += SIGNALS["taxonomic-pairs with non-taxonomic"]
    if left.taxo and right.taxo:
        score += SIGNALS["both taxonomic"]
    for one, other in ((left, right), (right, left)):
        if one.thing is not None and one.thing & other.words:
            score += SIGNALS["the taxonomy's subject is in the other"]
    score += SIGNALS["generality, held back"] * (left.general + right.general)
    return score


def _pairs(memory) -> list:
    """Every merge available now, best first."""
    nodes = memory["nodes"]
    keys = sorted(nodes)
    out = []
    for index, a in enumerate(keys):
        for b in keys[index + 1:]:
            out.append((affinity(nodes[a], nodes[b]), a, b))
    return sorted(out, key=lambda one: (-one[0], one[1], one[2]))


def _merge(memory, a: str, b: str, why: str) -> str:
    nodes = memory["nodes"]
    left, right = nodes.pop(a), nodes.pop(b)
    memory["made"] = made = memory.get("made", 0) + 1
    name = f"int{made}"
    nodes[name] = Node(name=name, leaves=left.leaves | right.leaves,
                       words=left.words | right.words,
                       taxo=left.taxo or right.taxo, thing=None,
                       general=min(left.general, right.general))
    memory["steps"] = memory.get("steps", []) + [(a, b, name, why)]
    memory["nodes"] = nodes
    if len(nodes) == 1:
        memory["tree"] = True
    return CONTINUE


def operators() -> list:
    """The four ways two nodes can belong together, in the order measured.

    Every one repeats: a proof is many steps and each is a fresh choice.
    A condition reads `nodes` and never takes from it -- `proposes` runs
    every cycle, so one that consumed what it tested would spend it before
    the action ran (the rule V3 learned the hard way).
    """
    def grounding(memory):
        """The best pair where one side's taxonomy is about the other."""
        nodes = memory.get("nodes", {})
        for score, a, b in _pairs(memory):
            if score <= 0:
                break
            for this, that in ((nodes[a], nodes[b]), (nodes[b], nodes[a])):
                if this.thing is not None and this.thing & that.words:
                    return a, b
        return None

    def elaborating(memory):
        nodes = memory.get("nodes", {})
        for score, a, b in _pairs(memory):
            if score <= 0:
                break
            if nodes[a].taxo and nodes[b].taxo:
                return a, b
        return None

    def joining(memory):
        for score, a, b in _pairs(memory):
            if score > 0:
                return a, b
        return None

    def anything(memory):
        pairs = _pairs(memory)
        return (pairs[0][1], pairs[0][2]) if pairs else None

    def doing(find, why):
        """An operator that fires only where its own pattern is there.

        A repeating operator whose condition outlives its action fires for
        ever -- V2 hung the v688 suite for two hours that way -- so the
        condition is the pattern itself, not `there is still work`. It
        reads `nodes` and never takes from it, because `proposes` runs
        every cycle and a condition that consumed what it tested would
        spend it before the action ran.
        """
        def proposes(memory) -> bool:
            return (len(memory.get("nodes", {})) > 1
                    and find(memory) is not None)

        def apply(memory):
            found = find(memory)
            if found is None:
                return DECLINED
            return _merge(memory, found[0], found[1], why)

        return proposes, apply

    ground_if, ground = doing(grounding, "ground")
    elaborate_if, elaborate = doing(elaborating, "elaborate")
    join_if, join = doing(joining, "join")
    apply_if, apply_rule = doing(anything, "apply")

    def answer(memory):
        memory["proof"] = as_proof(memory["steps"])
        return ANSWERED

    return [
        Operator(name="ground", apply=ground, proposes=ground_if,
                 rule="a taxonomic sentence and the thing it grounds",
                 needs=("nodes",), gives=("nodes",), repeats=True),
        Operator(name="elaborate", apply=elaborate, proposes=elaborate_if,
                 rule="two taxonomic sentences are one step",
                 needs=("nodes",), gives=("nodes",), repeats=True),
        Operator(name="join", apply=join, proposes=join_if,
                 rule="the two nodes sharing most words",
                 needs=("nodes",), gives=("nodes",), repeats=True),
        Operator(name="apply", apply=apply_rule, proposes=apply_if,
                 rule="what is left: the general rule, applied last",
                 needs=("nodes",), gives=("nodes",), repeats=True),
        Operator(name="the tree", apply=answer, needs=("tree",),
                 rule="one node left, so the proof is built",
                 gives=("proof",)),
    ]


def as_proof(steps: list) -> str:
    parts = []
    for index, (a, b, name, _) in enumerate(steps):
        head = "hypothesis" if index == len(steps) - 1 else name
        parts.append(f"{a} & {b} -> {head}")
    return "; ".join(parts) + "; " if parts else ""


def build(sentences: dict, hypothesis: str, executive=None) -> tuple:
    """(proof string, trace). `sentences` is id -> text."""
    ids = sorted(sentences, key=lambda s: int(s.replace("sent", "")))
    if len(ids) <= 1:
        return (f"{ids[0]} -> hypothesis; " if ids else ""), None
    hyp = words(hypothesis)
    nodes = {}
    for one in ids:
        thing, _ = taxonomy(sentences[one])
        nodes[one] = Node(name=one, leaves=frozenset({one}),
                          words=words(sentences[one]),
                          taxo=bool(TAXONOMIC.search(sentences[one].lower())),
                          thing=thing,
                          general=generality(sentences[one], hyp))
    memory = Working({"nodes": nodes, "made": 0, "steps": []},
                     goal=f"prove: {hypothesis}")
    executive = executive or Executive(operators(), name="entailment")
    trace = executive.run(memory)
    return memory.get("proof", ""), trace


# -- the data --------------------------------------------------------------

def load(task: str = "task_1", split: str = "dev") -> list:
    path = DATASET / task / f"{split}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# -- the baselines anything has to beat ------------------------------------

def one_step(sentences: list, texts=None, hypothesis="") -> str:
    return " & ".join(sentences) + " -> hypothesis; "


def chain(sentences: list, texts=None, hypothesis="") -> str:
    if len(sentences) <= 1:
        return f"{sentences[0]} -> hypothesis; " if sentences else ""
    parts, carry = [], sentences[0]
    for index, one in enumerate(sentences[1:], start=1):
        head = "hypothesis" if index == len(sentences) - 1 else f"int{index}"
        parts.append(f"{carry} & {one} -> {head}")
        carry = head
    return "; ".join(parts) + "; "


def by_executive(sentences: list, texts=None, hypothesis="") -> str:
    proof, _ = build({one: texts[one] for one in sentences}, hypothesis)
    return proof


BASELINES = {"one-step": one_step, "chain": chain,
             "executive": by_executive}


@dataclass
class Score:
    hits: int = 0
    predicted: int = 0
    gold: int = 0
    exact: int = 0
    trees: int = 0
    by_depth: dict = field(default_factory=lambda: collections.defaultdict(
        lambda: [0, 0]))

    def add(self, got: set, want: set, depth: int) -> None:
        self.hits += len(got & want)
        self.predicted += len(got)
        self.gold += len(want)
        self.exact += got == want
        self.trees += 1
        bucket = self.by_depth[depth]
        bucket[0] += got == want
        bucket[1] += 1

    @property
    def f1(self) -> float:
        precision = self.hits / self.predicted if self.predicted else 0.0
        recall = self.hits / self.gold if self.gold else 0.0
        return (2 * precision * recall / (precision + recall)
                if precision + recall else 0.0)

    @property
    def whole(self) -> float:
        return self.exact / self.trees if self.trees else 0.0


def measure(rows: list, build_one) -> tuple:
    """(all steps, inner steps) as two `Score`s."""
    whole, within = Score(), Score()
    for row in rows:
        sentences = sorted(row["meta"]["triples"],
                           key=lambda s: int(s.replace("sent", "")))
        want = signature(row["proof"])
        got = signature(build_one(sentences, row["meta"]["triples"],
                                  row["hypothesis"]))
        whole.add(got, want, row["depth_of_proof"])
        within.add(inner(got), inner(want), row["depth_of_proof"])
    return whole, within


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task_1")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--only", default="",
                        help="one baseline by name, or all of them")
    options = parser.parse_args(argv)

    rows = load(options.task, options.split)
    if not rows:
        print(f"no data at {DATASET / options.task}; "
              f"run `python -m regenerate --only entailmentbank`")
        return 2
    print(f"{options.task}/{options.split}: {len(rows)} trees\n")
    print(f"{'method':<12} {'steps F1':>9} {'whole tree':>11} "
          f"{'inner F1':>9} {'inner exact':>12}")
    print("-" * 60)
    for name, build_one in BASELINES.items():
        if options.only and name != options.only:
            continue
        whole, within = measure(rows, build_one)
        print(f"{name:<12} {whole.f1:>9.1%} {whole.whole:>11.1%} "
              f"{within.f1:>9.1%} {within.whole:>12.1%}")
        detail = "  ".join(f"d{d}:{b[0]}/{b[1]}"
                           for d, b in sorted(whole.by_depth.items()))
        print(f"{'':<12} by depth: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
