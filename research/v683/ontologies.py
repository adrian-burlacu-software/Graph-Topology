"""Which of these ontologies can actually be reasoned over?

    python -m research.v683.ontologies

`diagnose.py` showed that inference over v633's ConceptNet layer is either
explosive or over-pruned, and that no gate fixes it. That is a property of one
substrate, not a verdict on the idea. This scores four candidates on the same
label-free metrics so the choice can be made on evidence.

The metrics are deliberately not "accuracy". There are no labels. Every metric
here is a structural property that can be computed from the data alone, and
each one corresponds to something that has to be true before reasoning is even
well-defined:

    acyclic          "more general than" is a partial order, or it is nothing
    connected        two concepts in different components cannot be compared
    determinate      how many ways there are to generalise a concept
    bounded          how many ancestors a concept has -- the explosion metric
    checkable        whether the source ships constraints that can catch an
                     error without a human, e.g. owl:disjointWith
    contradiction    derived statements that violate those constraints
    grounded         how many concepts carry a non-taxonomic fact at all

A source that is cyclic and fragmented cannot support "what makes X more
general than Y" no matter how many facts it has.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sqlite3
import statistics
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import normalize, substrate

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "v683"
YAGO_DIRECTORY = REPOSITORY_ROOT / "data" / "yago"
ASCENT_CSV = REPOSITORY_ROOT / "data" / "ascentpp.csv"

csv.field_size_limit(10 ** 7)


@dataclass
class Ontology:
    """A taxonomy plus the non-taxonomic facts hanging off it."""

    name: str
    parents: dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    facts: dict[str, set[tuple[str, str]]] = field(default_factory=lambda: collections.defaultdict(set))
    disjoint: dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> set[str]:
        return set(self.parents) | {p for v in self.parents.values() for p in v}


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_wordnet(database: Path, repair: bool = True) -> Ontology:
    """WordNet: hand-curated, sense-disambiguated, taxonomy-only.

    With `repair`, mutual subsumptions are dropped. There is exactly one in
    97,666 edges -- `restrain.v.01` and `inhibit.v.04` are asserted as both
    `is_a` and `has_subtype` in the same direction -- and it is the sole reason
    the raw load is cyclic. Neither direction can be trusted once they
    contradict, so both go. Pass `repair=False` to see the unrepaired graph.
    """
    ontology = Ontology("wordnet")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        for child, parent in connection.execute(
            "SELECT subject, object FROM edges WHERE relation='is_a' "
            "AND subject LIKE 'wn:%' AND object LIKE 'wn:%'"
        ):
            ontology.parents[child].add(parent)
        for parent, child in connection.execute(
            "SELECT subject, object FROM edges WHERE relation='has_subtype' "
            "AND subject LIKE 'wn:%' AND object LIKE 'wn:%'"
        ):
            ontology.parents[child].add(parent)
        if repair:
            mutual = {
                (child, parent)
                for child, above in ontology.parents.items()
                for parent in above
                if child in ontology.parents.get(parent, ())
            }
            for child, parent in mutual:
                ontology.parents[child].discard(parent)
            ontology.notes["repaired_mutual_subsumptions"] = len(mutual) // 2
        placeholders = ",".join("?" * len(substrate.ATTRIBUTE_RELATIONS))
        for subject, relation, obj in connection.execute(
            f"SELECT subject, relation, object FROM edges WHERE subject LIKE 'wn:%' "
            f"AND relation IN ({placeholders})", substrate.ATTRIBUTE_RELATIONS
        ):
            if relation not in ("is_a", "has_subtype"):
                ontology.facts[subject].add((relation, obj))
    finally:
        connection.close()
    ontology.notes.update({"sense_disambiguated": True, "confidence_scores": False,
                           "ships_constraints": False})
    return ontology


def load_conceptnet(database: Path) -> Ontology:
    """ConceptNet: the current substrate, kept as the baseline to beat."""
    ontology = Ontology("conceptnet")
    corpus = substrate.load(database, substrate.ATTRIBUTE_RELATIONS, "attributes",
                            normalization=normalize.SAFE)
    for concept, predicates in corpus.items:
        if not concept.startswith("en:"):
            continue
        for relation, obj in predicates:
            if relation == "is_a":
                if obj.startswith("en:"):
                    ontology.parents[concept].add(obj)
            else:
                ontology.facts[concept].add((relation, obj))
    ontology.notes = {"sense_disambiguated": False, "confidence_scores": False,
                      "ships_constraints": False}
    return ontology


def _yago_lines(directory: Path, member: str) -> Iterator[str]:
    """Stream a YAGO turtle file straight out of its zip."""
    archive = directory / f"yago-4.6-{member}.zip"
    extracted = directory / "x" / f"yago-{member}.ttl"
    if extracted.is_file():
        with extracted.open(encoding="utf-8", errors="ignore") as handle:
            yield from handle
        return
    with zipfile.ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith(".ttl"))
        with bundle.open(name) as handle:
            for raw in handle:
                yield raw.decode("utf-8", "ignore")


_DISJOINT = re.compile(r"^(\S+)\s+a\s+sh:NodeShape")


def load_yago(directory: Path) -> Ontology:
    """YAGO 4.6: Wikidata cleaned against a schema.org upper taxonomy.

    The taxonomy file is tab-separated triples after a prefix block. The schema
    file is block turtle and carries the part nothing else has -- owl:disjointWith
    between top classes, which makes a whole family of errors machine-checkable.
    """
    ontology = Ontology("yago")
    for line in _yago_lines(directory, "taxonomy"):
        if "rdfs:subClassOf" not in line:
            continue
        parts = line.rstrip(" .\n").split("\t")
        if len(parts) >= 3:
            ontology.parents[parts[0].strip()].add(parts[2].strip())

    current = None
    for line in _yago_lines(directory, "schema"):
        stripped = line.strip()
        match = _DISJOINT.match(stripped)
        if match:
            current = match.group(1)
        elif current and stripped.startswith("owl:disjointWith"):
            targets = stripped[len("owl:disjointWith"):].rstrip(";. ").split(",")
            for target in targets:
                target = target.strip()
                if target:
                    ontology.disjoint[current].add(target)
                    ontology.disjoint[target].add(current)
        elif stripped.endswith("."):
            current = None
    ontology.notes = {"sense_disambiguated": True, "confidence_scores": False,
                      "ships_constraints": True,
                      "disjoint_pairs": sum(len(v) for v in ontology.disjoint.values()) // 2}
    return ontology


def load_ascentpp(path: Path, min_typicality: float = 0.0) -> Ontology:
    """Ascent++: a property layer with graded scores, and barely a taxonomy."""
    ontology = Ontology("ascentpp")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["typicality"]) < min_typicality:
                continue
            subject, relation, tail = row["subject"], row["relation"], row["tail"]
            if relation == "IsA":
                ontology.parents[subject].add(tail)
            else:
                ontology.facts[subject].add((relation, tail))
    ontology.notes = {"sense_disambiguated": False, "confidence_scores": True,
                      "ships_constraints": False,
                      "subgroups": "lexical, not word senses"}
    return ontology


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def find_cycles(parents: dict[str, set[str]], limit: int = 5) -> list[list[str]]:
    colour: dict[str, int] = collections.defaultdict(int)
    cycles: list[list[str]] = []
    for start in list(parents):
        if colour[start]:
            continue
        colour[start] = 1
        stack = [(start, iter(parents.get(start, ())))]
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
                if len(cycles) < limit:
                    cycles.append(path[path.index(nxt):] + [nxt])
            elif colour[nxt] == 0:
                colour[nxt] = 1
                path.append(nxt)
                stack.append((nxt, iter(parents.get(nxt, ()))))
    return cycles


def components(parents: dict[str, set[str]], nodes: set[str]) -> list[int]:
    adjacency: dict[str, set[str]] = collections.defaultdict(set)
    for child, above in parents.items():
        for parent in above:
            adjacency[child].add(parent)
            adjacency[parent].add(child)
    seen: set[str] = set()
    sizes: list[int] = []
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
        sizes.append(size)
    return sorted(sizes, reverse=True)


def ancestors(parents: dict[str, set[str]], node: str, limit: int = 20) -> set[str]:
    seen = {node}
    frontier = {node}
    for _ in range(limit):
        nxt: set[str] = set()
        for current in frontier:
            nxt |= parents.get(current, set()) - seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen - {node}


def contradiction_rate(ontology: Ontology, sample: list[str]) -> dict[str, Any]:
    """Ancestors that the source's own constraints say cannot co-occur.

    Only YAGO ships disjointness, so only YAGO can be checked this way. That
    asymmetry is the finding, not a gap in the measurement: the other sources
    provide no way to be caught being wrong.
    """
    if not ontology.disjoint:
        return {"checkable": False,
                "reason": "source ships no disjointness or domain constraints"}
    violations = 0
    examples: list[str] = []
    for node in sample:
        above = ancestors(ontology.parents, node)
        for concept in above:
            clash = ontology.disjoint.get(concept, set()) & above
            if clash:
                violations += 1
                if len(examples) < 5:
                    examples.append(f"{node}: {concept} vs {sorted(clash)[0]}")
                break
    return {"checkable": True, "sampled": len(sample), "violations": violations,
            "rate": round(violations / len(sample), 5) if sample else 0.0,
            "examples": examples}


def score(ontology: Ontology, sample_size: int, seed: int) -> dict[str, Any]:
    import random

    nodes = ontology.nodes
    cycles = find_cycles(ontology.parents)
    sizes = components(ontology.parents, nodes)
    rng = random.Random(seed)
    population = sorted(ontology.parents)
    sample = rng.sample(population, min(sample_size, len(population))) if population else []
    counts = [len(ancestors(ontology.parents, node)) for node in sample]
    parent_counts = [len(v) for v in ontology.parents.values()]
    grounded = sum(1 for node in nodes if ontology.facts.get(node))
    return {
        "source": ontology.name,
        "nodes": len(nodes),
        "taxonomy_edges": sum(len(v) for v in ontology.parents.values()),
        "fact_edges": sum(len(v) for v in ontology.facts.values()),
        "acyclic": not cycles,
        "cycles_found": len(cycles),
        "cycle_example": [n.split(":")[-1] for n in cycles[0][:6]] if cycles else [],
        "components": len(sizes),
        "largest_component_share": round(sizes[0] / len(nodes), 4) if nodes else 0.0,
        "mean_parents": round(statistics.fmean(parent_counts), 3) if parent_counts else 0.0,
        "multi_parent_share": round(
            sum(1 for c in parent_counts if c > 1) / len(parent_counts), 4
        ) if parent_counts else 0.0,
        "ancestors_median": statistics.median(counts) if counts else 0,
        "ancestors_mean": round(statistics.fmean(counts), 1) if counts else 0.0,
        "ancestors_max": max(counts) if counts else 0,
        "grounded_share": round(grounded / len(nodes), 4) if nodes else 0.0,
        "contradictions": contradiction_rate(ontology, sample),
        **ontology.notes,
    }


def run(database: Path, yago: Path, ascent: Path, output: Path,
        sample_size: int, seed: int) -> dict[str, Any]:
    results = []
    for name, loader in (
        ("wordnet", lambda: load_wordnet(database)),
        ("conceptnet", lambda: load_conceptnet(database)),
        ("yago", lambda: load_yago(yago)),
        ("ascentpp", lambda: load_ascentpp(ascent)),
    ):
        try:
            results.append(score(loader(), sample_size, seed))
        except (FileNotFoundError, StopIteration, zipfile.BadZipFile) as error:
            results.append({"source": name, "unavailable": str(error)})
    report = {"results": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "v683_ontologies.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def render(report: dict[str, Any]) -> str:
    rows = [r for r in report["results"] if "unavailable" not in r]
    lines = [
        "# Which ontology can be reasoned over?",
        "",
        "| source | nodes | taxonomy edges | facts | acyclic | components | largest | mean parents | ancestors med/mean/max | grounded |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        acyclic = "yes" if row["acyclic"] else f"NO ({row['cycles_found']})"
        depth = (f"{row['ancestors_median']:.0f} / {row['ancestors_mean']} "
                 f"/ {row['ancestors_max']:,}")
        lines.append(
            f"| {row['source']} | {row['nodes']:,} | {row['taxonomy_edges']:,} "
            f"| {row['fact_edges']:,} | {acyclic} "
            f"| {row['components']:,} | {row['largest_component_share']:.1%} "
            f"| {row['mean_parents']} | {depth} "
            f"| {row['grounded_share']:.1%} |"
        )
    lines += ["", "## Machine-checkable errors", ""]
    for row in rows:
        check = row["contradictions"]
        if check.get("checkable"):
            lines.append(
                f"- **{row['source']}**: {check['violations']:,} of "
                f"{check['sampled']:,} sampled concepts inherit from classes the "
                f"schema declares disjoint ({check['rate']:.3%})."
            )
            for example in check["examples"][:3]:
                lines.append(f"      {example}")
        else:
            lines.append(f"- **{row['source']}**: not checkable — {check['reason']}.")
    lines += ["", "## Capabilities", "",
              "| source | senses disambiguated | confidence scores | ships constraints |",
              "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['source']} | {'yes' if row.get('sense_disambiguated') else 'no'} "
            f"| {'yes' if row.get('confidence_scores') else 'no'} "
            f"| {'yes' if row.get('ships_constraints') else 'no'} |"
        )
    missing = [r for r in report["results"] if "unavailable" in r]
    if missing:
        lines += ["", "## Unavailable", ""]
        for row in missing:
            lines.append(f"- {row['source']}: {row['unavailable']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database", type=Path, default=substrate.DEFAULT_DATABASE)
    parser.add_argument("--yago", type=Path, default=YAGO_DIRECTORY)
    parser.add_argument("--ascent", type=Path, default=ASCENT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    report = run(arguments.database, arguments.yago, arguments.ascent,
                 arguments.output, arguments.sample, arguments.seed)
    text = render(report)
    (arguments.output / "v683_ontologies.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
