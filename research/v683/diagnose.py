"""Is the normalized graph a usable basis for answering general questions?

    python -m research.v683.diagnose

This is the evidence behind that question, kept runnable so the claims can be
rechecked rather than believed. It answers "what would break", not "how well
does it work" -- there is no ground truth here for the latter, and that absence
is itself the main finding.

Four probes, in the order the answer depends on them:

1. Is the taxonomy navigable?      cycles, fragmentation, hubs
2. Do properties inherit?          direct facts against derived facts
3. Can bad inheritance be gated?   three candidate gates, all measured
4. Can edges be cross-validated?   ConceptNet is_a against WordNet's curated tree

Probe 3 is the one that matters, and all three gates fail it. Read `VERDICT` at
the bottom of the output before building anything on top of this.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from . import normalize, substrate

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "v683"

PROBE_CONCEPTS = ("en:dog", "en:hammer", "en:violin")
PROBE_EDGES = (
    ("en:dog", "en:mammal"), ("en:dog", "en:thing"),
    ("en:hammer", "en:tool"), ("en:hammer", "en:sports equipment"),
    ("en:hammer", "en:match"), ("en:violin", "en:stringed instrument"),
)


class Graph:
    """The normalized attribute graph plus WordNet's curated hypernymy."""

    def __init__(self, database: Path, normalization: normalize.Normalization):
        corpus = substrate.load(database, substrate.ATTRIBUTE_RELATIONS,
                                "attributes", normalization=normalization)
        self.predicates = {name: set(values) for name, values in corpus.items}
        self.size = len(self.predicates)
        self.parents: dict[str, set[str]] = collections.defaultdict(set)
        self.extension: dict[str, set[str]] = collections.defaultdict(set)
        for name, values in self.predicates.items():
            for relation, obj in values:
                self.extension[obj].add(name)
                if relation == "is_a":
                    self.parents[name].add(obj)

        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            self.hypernym: dict[str, set[str]] = collections.defaultdict(set)
            for child, parent in connection.execute(
                "SELECT subject, object FROM edges WHERE relation='is_a' "
                "AND subject LIKE 'wn:%' AND object LIKE 'wn:%'"
            ):
                self.hypernym[child].add(parent)
            for parent, child in connection.execute(
                "SELECT subject, object FROM edges WHERE relation='has_subtype' "
                "AND subject LIKE 'wn:%'"
            ):
                self.hypernym[child].add(parent)
            self.senses: dict[str, set[str]] = collections.defaultdict(set)
            for lemma, synset in connection.execute(
                "SELECT subject, object FROM edges WHERE relation='has_sense'"
            ):
                self.senses[lemma].add(synset)
        finally:
            connection.close()
        self._ancestors: dict[str, frozenset[str]] = {}

    def facts(self, concept: str) -> set[tuple[str, str]]:
        """Predicates that assert something other than membership."""
        return {(r, o) for r, o in self.predicates.get(concept, ()) if r != "is_a"}

    def information(self, concept: str) -> float:
        """-log2 P(concept) over how many individuals point at it."""
        size = len(self.extension.get(concept, ()))
        return -math.log2(size / self.size) if size else math.log2(self.size)

    def agreement(self, left: str, right: str) -> float:
        """Jaccard overlap of what two concepts point at."""
        a = {o for _, o in self.predicates.get(left, ())}
        b = {o for _, o in self.predicates.get(right, ())}
        return len(a & b) / len(a | b) if (a | b) else 0.0

    def ancestors(self, synset: str) -> frozenset[str]:
        if synset in self._ancestors:
            return self._ancestors[synset]
        seen, frontier = {synset}, {synset}
        for _ in range(12):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self.hypernym.get(node, set()) - seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        self._ancestors[synset] = frozenset(seen)
        return self._ancestors[synset]

    def justifying_senses(self, child: str, parent: str) -> frozenset[str] | None:
        """Senses of `child` that make `child is_a parent` true in WordNet.

        None means WordNet cannot judge -- either side may lack senses -- which
        is not the same as the edge being wrong.
        """
        targets = self.senses.get(parent, set())
        if not targets:
            return None
        found = {s for s in self.senses.get(child, set())
                 if self.ancestors(s) & targets}
        return frozenset(found) if found else None

    def inherit(self, concept: str, depth: int = 4, gate: str = "none",
                min_bits: float = 0.0, min_agreement: float = 0.0
                ) -> list[tuple[int, str, str, str]]:
        """Walk is_a upward collecting facts, under one gating policy.

        Cycle-safe by construction: `is_a` contains cycles, so a visited set is
        not an optimization here, it is required for termination.
        """
        seen = {concept}
        frontier: list[tuple[str, frozenset[str] | None]] = [(concept, None)]
        gathered: list[tuple[int, str, str, str]] = []
        for level in range(1, depth + 1):
            nxt: list[tuple[str, frozenset[str] | None]] = []
            for node, scope in frontier:
                for parent in self.parents.get(node, ()):
                    if parent in seen:
                        continue
                    if gate == "information":
                        if self.information(parent) < min_bits:
                            continue
                    elif gate == "agreement":
                        if self.agreement(node, parent) < min_agreement:
                            continue
                    elif gate in ("confirmed", "sense"):
                        justifying = self.justifying_senses(node, parent)
                        if justifying is None:
                            continue
                        if gate == "sense" and scope is not None and not (
                            justifying & scope
                        ):
                            continue
                        seen.add(parent)
                        nxt.append((parent, justifying))
                        continue
                    seen.add(parent)
                    nxt.append((parent, None))
            for parent, _ in nxt:
                gathered += [(level, parent, r, o) for r, o in self.facts(parent)]
            frontier = nxt
            if not frontier:
                break
        return gathered


def probe_navigable(graph: Graph) -> dict[str, Any]:
    """1. Cycles and fragmentation decide whether 'more general' is defined."""
    colour: dict[str, int] = collections.defaultdict(int)
    cycles: list[list[str]] = []
    nodes = set(graph.parents) | {p for v in graph.parents.values() for p in v}
    for start in nodes:
        if colour[start]:
            continue
        colour[start] = 1
        stack = [(start, iter(graph.parents.get(start, ())))]
        path = [start]
        while stack:
            node, walk = stack[-1]
            nxt = next(walk, None)
            if nxt is None:
                colour[node] = 2
                stack.pop()
                path.pop()
                continue
            if colour[nxt] == 1:
                if len(cycles) < 5:
                    cycles.append(path[path.index(nxt):] + [nxt])
            elif colour[nxt] == 0:
                colour[nxt] = 1
                path.append(nxt)
                stack.append((nxt, iter(graph.parents.get(nxt, ()))))

    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    for child, parents in graph.parents.items():
        for parent in parents:
            adjacency[child].add(parent)
            adjacency[parent].add(child)
    seen: set[str] = set()
    components: list[int] = []
    for node in nodes:
        if node in seen:
            continue
        queue, size = [node], 0
        seen.add(node)
        while queue:
            current = queue.pop()
            size += 1
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(size)
    components.sort(reverse=True)
    return {
        "nodes": len(nodes),
        "is_a_edges": sum(len(v) for v in graph.parents.values()),
        "cycles_found": len(cycles),
        "cycle_examples": [[n.replace("en:", "") for n in c[:6]] for c in cycles],
        "components": len(components),
        "largest_component": components[0],
        "largest_share": round(components[0] / len(nodes), 4),
        "second_component": components[1] if len(components) > 1 else 0,
    }


def probe_inheritance(graph: Graph, sample_size: int, seed: int) -> dict[str, Any]:
    """2. What inheritance yields, and on which concepts it goes wrong."""
    examples = {}
    for concept in PROBE_CONCEPTS:
        first = [(p.replace("en:", ""), r, o.replace("en:", ""))
                 for level, p, r, o in graph.inherit(concept, 1) if level == 1]
        examples[concept] = {
            "direct_facts": len(graph.facts(concept)),
            "inherited_depth_1": len(first),
            "sources": sorted({item[0] for item in first}),
            "sample": first[:6],
        }
    rng = random.Random(seed)
    population = [c for c in graph.predicates if graph.parents.get(c)]
    sample = rng.sample(population, min(sample_size, len(population)))
    direct = [len(graph.facts(c)) for c in sample]
    derived = [len(graph.inherit(c, 4)) for c in sample]
    return {
        "examples": examples,
        "sample": len(sample),
        "direct_median": statistics.median(direct),
        "direct_mean": round(statistics.fmean(direct), 3),
        "derived_median": statistics.median(derived),
        "derived_mean": round(statistics.fmean(derived), 1),
        "derived_max": max(derived),
        "concepts_with_no_direct_facts": round(
            sum(1 for d in direct if d == 0) / len(direct), 4
        ),
    }


def probe_gates(graph: Graph, sample_size: int, seed: int) -> dict[str, Any]:
    """3. The decisive probe: can bad inheritance be filtered out?"""
    rng = random.Random(seed)
    population = [c for c in graph.predicates if graph.parents.get(c)]
    sample = rng.sample(population, min(sample_size, len(population)))
    barren = [c for c in sample if not graph.facts(c)]
    policies = (
        ("none", {}),
        ("information>=8bits", {"gate": "information", "min_bits": 8.0}),
        ("agreement>=0.02", {"gate": "agreement", "min_agreement": 0.02}),
        ("wordnet_confirmed", {"gate": "confirmed"}),
        ("sense_scoped", {"gate": "sense"}),
    )
    rows = []
    for label, options in policies:
        counts = [len(graph.inherit(c, 4, **options)) for c in sample]
        covered = sum(1 for c in barren if graph.inherit(c, 4, **options))
        rows.append({
            "gate": label,
            "median": statistics.median(counts),
            "mean": round(statistics.fmean(counts), 1),
            "p99": sorted(counts)[int(len(counts) * 0.99)],
            "max": max(counts),
            "barren_concepts_covered": round(covered / len(barren), 4) if barren else 0.0,
        })
    violin = {
        label: sorted({p.replace("en:", "")
                       for _, p, _, _ in graph.inherit("en:violin", 2, **options)})
        for label, options in policies
    }
    return {"rows": rows, "violin_sources_depth_2": violin}


def probe_cross_validation(graph: Graph, database: Path, sample_size: int,
                           seed: int) -> dict[str, Any]:
    """4. WordNet as an independent judge of ConceptNet's taxonomy."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        edges = list(connection.execute(
            "SELECT subject, object FROM edges WHERE relation='is_a' "
            "AND subject LIKE 'en:%' AND object LIKE 'en:%'"
        ))
    finally:
        connection.close()
    rng = random.Random(seed)
    sample = rng.sample(edges, min(sample_size, len(edges)))
    tally = collections.Counter(
        "confirmed" if graph.justifying_senses(a, b)
        else ("unjudgeable" if not graph.senses.get(b) or not graph.senses.get(a)
              else "contradicted")
        for a, b in sample
    )
    judged = tally["confirmed"] + tally["contradicted"]
    return {
        "total_edges": len(edges),
        "sampled": len(sample),
        "confirmed": tally["confirmed"],
        "contradicted": tally["contradicted"],
        "unjudgeable": tally["unjudgeable"],
        "precision_among_judgeable": round(tally["confirmed"] / judged, 4) if judged else None,
        "probe_edges": {
            f"{a} is_a {b}": bool(graph.justifying_senses(a, b))
            for a, b in PROBE_EDGES
        },
    }


VERDICT = """\
The graph supports direct, provenance-carrying lookup over 3.9M edges. It does
not yet support multi-hop inference, and no gate tested here fixes that.

Inheritance is either explosive or over-pruned, with nothing usable between:
trusting every is_a edge yields a mean of ~1,550 derived facts per concept and
a maximum near 13,000, most of it wrong; sense-scoping cuts the mean to ~22 but
drops coverage of fact-less concepts from ~75% to ~14%, and it removes correct
parents (violin loses `musical instrument`) while keeping wrong ones (hammer
keeps `sports equipment`, correct only for the throwing sense).

Information content and profile agreement both fail outright. The database is
too sparse for any statistical agreement measure: `dog is_a mammal` scores
0.0034 agreement, below the junk it is meant to outrank.

What is missing is not another heuristic. It is a labelled set saying which
derived facts are true. Without one, every gate above is tuned against an
unmeasurable target -- which is exactly how V682's benchmark ended up asserting
`actual = expected` and reporting 1.0 accuracy.
"""


def run(database: Path, output: Path, sample_size: int, edge_sample: int,
        seed: int, normalization: normalize.Normalization) -> dict[str, Any]:
    graph = Graph(database, normalization)
    report = {
        "database": str(database.resolve()),
        "normalization": normalization.name,
        "individuals": graph.size,
        "navigable": probe_navigable(graph),
        "inheritance": probe_inheritance(graph, sample_size, seed),
        "gates": probe_gates(graph, sample_size, seed),
        "cross_validation": probe_cross_validation(graph, database, edge_sample, seed),
        "verdict": VERDICT,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "v683_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def render(report: dict[str, Any]) -> str:
    nav, inh = report["navigable"], report["inheritance"]
    lines = [
        "# V683 diagnosis: can this graph answer general questions?",
        "",
        f"`{report['database']}` under `{report['normalization']}`, "
        f"{report['individuals']:,} individuals.",
        "",
        "## 1. Is the taxonomy navigable?",
        "",
        f"- {nav['nodes']:,} nodes, {nav['is_a_edges']:,} `is_a` edges",
        f"- **cycles: {nav['cycles_found']}** — `is_a` depth is not a valid "
        f"generality measure",
        f"- {nav['components']:,} components; largest holds "
        f"{nav['largest_share']:.1%}, second is {nav['second_component']:,}",
        "",
    ]
    for cycle in nav["cycle_examples"][:3]:
        lines.append(f"      {' -> '.join(cycle)}")
    lines += ["", "## 2. Do properties inherit?", ""]
    for concept, data in inh["examples"].items():
        lines.append(
            f"- `{concept}`: {data['direct_facts']} direct, "
            f"{data['inherited_depth_1']} inherited at depth 1, "
            f"from {data['sources']}"
        )
    lines += [
        "",
        f"Over {inh['sample']:,} concepts: direct facts median "
        f"{inh['direct_median']:.0f} / mean {inh['direct_mean']}; derived median "
        f"{inh['derived_median']:.0f} / mean {inh['derived_mean']} / max "
        f"{inh['derived_max']:,}.",
        f"{inh['concepts_with_no_direct_facts']:.1%} of concepts have no direct "
        f"facts at all.",
        "",
        "## 3. Can bad inheritance be gated?",
        "",
        "| gate | median | mean | p99 | max | fact-less concepts covered |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["gates"]["rows"]:
        lines.append(
            f"| {row['gate']} | {row['median']:.0f} | {row['mean']} "
            f"| {row['p99']:,} | {row['max']:,} "
            f"| {row['barren_concepts_covered']:.1%} |"
        )
    lines += ["", "`en:violin` parents reached at depth 2, by gate:", ""]
    for gate, sources in report["gates"]["violin_sources_depth_2"].items():
        lines.append(f"- `{gate}`: {sources}")

    cross = report["cross_validation"]
    lines += [
        "",
        "## 4. Can edges be cross-validated against WordNet?",
        "",
        f"{cross['sampled']:,} ConceptNet `is_a` edges sampled of "
        f"{cross['total_edges']:,}: {cross['confirmed']:,} confirmed, "
        f"{cross['contradicted']:,} contradicted, {cross['unjudgeable']:,} "
        f"unjudgeable.",
        "",
        f"Precision among judgeable: "
        f"{cross['precision_among_judgeable']:.1%}"
        if cross["precision_among_judgeable"] else "",
        "",
    ]
    for edge, ok in cross["probe_edges"].items():
        lines.append(f"- `{edge}` → {'confirmed' if ok else 'REJECTED'}")
    lines += ["", "## Verdict", "", report["verdict"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database", type=Path, default=substrate.DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=1200)
    parser.add_argument("--edge-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--normalization", default="safe",
                        choices=sorted(normalize.NORMALIZATIONS))
    arguments = parser.parse_args()
    report = run(arguments.database, arguments.output, arguments.sample,
                 arguments.edge_sample, arguments.seed,
                 normalize.NORMALIZATIONS[arguments.normalization])
    text = render(report)
    (arguments.output / "v683_diagnosis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
