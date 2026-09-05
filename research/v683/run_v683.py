"""V683: does storing an ontology in a predicate trie minimize it?

    python -m research.v683.run_v683

Four claims from Burlacu & West (2021) are tested against the V633 semantic
graph (all WordNet plus all English ConceptNet). Each is stated so the data can
refuse it.

H1  Coverage ordering allocates fewer nodes than an arbitrary order.
    (Appendix 3.) Falsified if `global_coverage` does not beat `shuffled`.

H2  Branch-local ordering beats one global order.
    (Figure 20 against the Appendix 3 text.) Falsified if `adaptive_coverage`
    does not beat `global_coverage`.

H3  Access depth is set by the individual, not by the size of the vocabulary.
    (Sections 2.b and 15.) Falsified if mean depth climbs with the corpus.

H4  The greedy heuristic lands near the true optimum.
    Checked by exhaustive search over every global order on small samples.

Results are written to results/v683/ as JSON and as a readable report.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import measure, ordering, substrate
from .substrate import PAPER_TABLE_1, SLICES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "v683"

#: Figure 20 of the paper, as predicate paths.
FIGURE_20 = {
    "Saint Bernard": ("Big", "Long Hair", "Loud"),
    "Briard": ("Big", "Long Hair"),
    "Dalmatian": ("Big", "Spots"),
    "Border Collie": ("Big",),
    "Jack Russell Terrier": ("Spots",),
}


def _pct(value: float) -> str:
    return f"{value:.2%}"


def paper_reproduction() -> dict[str, Any]:
    """Table 1 must yield Figure 20, or nothing below is worth reading."""
    plans = {
        "adaptive_coverage": dict(ordering.adaptive_coverage(PAPER_TABLE_1)),
        "global_coverage": dict(ordering.global_coverage(PAPER_TABLE_1)),
        "optimal": dict(ordering.optimal(PAPER_TABLE_1)),
    }
    return {
        "table_1_coverage": dict(ordering.coverage(PAPER_TABLE_1)),
        "figure_20_expected": {k: list(v) for k, v in FIGURE_20.items()},
        "figure_20_reproduced": {
            name: plan == FIGURE_20 for name, plan in plans.items()
        },
        "nodes": {
            name: measure.measure(PAPER_TABLE_1, name, list(plan.items())).nodes
            for name, plan in plans.items()
        },
        "note": (
            "All three agree here at five nodes, which exhaustive search confirms "
            "is minimal. The paper's own example cannot separate its global "
            "description from its branch-local figure, so H2 needs real data."
        ),
    }


def ordering_comparison(corpus: substrate.Corpus) -> dict[str, Any]:
    """Score every ordering on one corpus, resolving H1 and H2."""
    results, timings = [], {}
    for name, function in ordering.ORDERINGS.items():
        started = time.perf_counter()
        plan = function(corpus)
        timings[name] = round(time.perf_counter() - started, 3)
        results.append(measure.measure(corpus, name, plan))
    results.sort(key=lambda item: item.nodes)
    scored = {item.ordering: item for item in results}
    arbitrary, glob = scored["shuffled"], scored["global_coverage"]
    adaptive, anti = scored["adaptive_coverage"], scored["anti_coverage"]
    return {
        "corpus": corpus.name,
        "individuals": len(corpus),
        "flat_cells": corpus.cells,
        "distinct_predicates": corpus.predicates,
        "measurements": [item.as_dict() for item in results],
        "seconds": timings,
        "h1_coverage_beats_arbitrary": {
            "nodes_saved": arbitrary.nodes - glob.nodes,
            "relative": round((arbitrary.nodes - glob.nodes) / arbitrary.nodes, 6),
            "holds": glob.nodes < arbitrary.nodes,
        },
        "h2_branch_local_beats_global": {
            "nodes_saved": glob.nodes - adaptive.nodes,
            "relative": round((glob.nodes - adaptive.nodes) / glob.nodes, 6),
            "holds": adaptive.nodes < glob.nodes,
        },
        "ordering_spread": {
            "best": results[0].ordering,
            "worst": results[-1].ordering,
            "nodes": anti.nodes - adaptive.nodes,
            "note": "How much of the stored size ordering alone decides.",
        },
    }


def scaling(corpus: substrate.Corpus, steps: int) -> dict[str, Any]:
    """H3: grow the vocabulary and watch depth against allocation.

    Judged on median depth. The mean is not usable here: `en:person` alone
    carries 6,231 predicates, so whether that one concept has been drawn yet
    moves the mean more than doubling the vocabulary does.
    """
    curve = measure.scaling_curve(corpus, ordering.global_coverage, steps=steps)
    first, last = curve[0], curve[-1]
    drift = last.median_depth / first.median_depth if first.median_depth else 0.0
    mean_drift = last.mean_depth / first.mean_depth if first.mean_depth else 0.0
    return {
        "curve": [asdict(point) for point in curve],
        "vocabulary_growth": round(last.individuals / first.individuals, 4),
        "node_growth": round(last.nodes / first.nodes, 4) if first.nodes else 0.0,
        "median_depth_growth": round(drift, 4),
        "mean_depth_growth": round(mean_drift, 4),
        "h3_depth_independent_of_vocabulary": {
            "first_median_depth": first.median_depth,
            "last_median_depth": last.median_depth,
            "first_mean_depth": first.mean_depth,
            "last_mean_depth": last.mean_depth,
            "holds": abs(drift - 1.0) < 0.10,
            "criterion": (
                "median access depth changes by less than 10% while the "
                "vocabulary grows by the factor reported above"
            ),
        },
    }


def optimality(corpus: substrate.Corpus, samples: int, width: int,
               seed: int) -> dict[str, Any]:
    """H4: how much of the achievable saving does the heuristic actually take?

    Draws individuals that share a small predicate universe, so every global
    order can be enumerated and the heuristics compared against the true best.
    """
    rng = random.Random(seed)
    by_predicate: dict[Any, list[int]] = {}
    for position, (_, predicates) in enumerate(corpus):
        for predicate in predicates:
            by_predicate.setdefault(predicate, []).append(position)
    anchors = [p for p, holders in by_predicate.items() if len(holders) >= 8]
    if not anchors:
        return {"samples": 0, "note": "no predicate is shared by enough individuals"}

    gaps: list[int] = []
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < samples and attempts < samples * 40:
        attempts += 1
        holders = by_predicate[rng.choice(anchors)]
        universe: set = set()
        chosen: list[tuple[str, frozenset]] = []
        for position in rng.sample(holders, min(10, len(holders))):
            individual, predicates = corpus[position]
            if len(universe | set(predicates)) > width:
                continue
            universe |= set(predicates)
            chosen.append((individual, frozenset(predicates)))
        if len(chosen) < 4 or len(universe) < 3:
            continue
        sample = substrate.Corpus("sample", tuple(chosen))
        best = measure.measure(sample, "optimal", ordering.optimal(sample, width)).nodes
        scores = {
            name: measure.measure(sample, name, function(sample)).nodes
            for name, function in ordering.ORDERINGS.items()
        }
        gaps.append(scores["global_coverage"] - best)
        records.append({
            "individuals": len(chosen), "predicates": len(universe),
            "optimal": best, **scores,
        })
    if not records:
        return {"samples": 0, "note": "no sample fit the predicate-width budget"}
    exact = sum(1 for gap in gaps if gap == 0)
    return {
        "samples": len(records),
        "max_predicate_universe": width,
        "global_coverage_hits_optimum": exact,
        "hit_rate": round(exact / len(records), 4),
        "mean_excess_nodes": round(statistics.fmean(gaps), 4),
        "max_excess_nodes": max(gaps),
        "h4_greedy_is_near_optimal": {
            "holds": statistics.fmean(gaps) < 0.5,
            "criterion": "mean excess over the exhaustive optimum below 0.5 nodes",
        },
        "records": records[:20],
    }


def run(database: Path, output: Path, steps: int, samples: int, width: int,
        seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "database": str(database.resolve()),
        "paper_reproduction": paper_reproduction(),
        "slices": {},
    }
    for name, relations in SLICES.items():
        corpus = substrate.load(database, relations, name)
        report["slices"][name] = ordering_comparison(corpus)
        if name == "attributes":
            report["scaling"] = scaling(corpus, steps)
            report["optimality"] = optimality(corpus, samples, width, seed)
            multi = corpus.filter_min_predicates(2)
            report["slices"]["attributes_multi_predicate"] = ordering_comparison(multi)
    _write(report, output)
    return report


def _write(report: dict[str, Any], output: Path) -> None:
    (output / "v683_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "v683_report.md").write_text(render(report), encoding="utf-8")


def render(report: dict[str, Any]) -> str:
    lines = [
        "# V683: ontology minimization by predicate trie",
        "",
        f"Database: `{report['database']}`",
        "",
    ]

    paper = report["paper_reproduction"]
    lines += [
        "## Appendix 3 reproduction",
        "",
        "| ordering | reproduces Figure 20 | nodes |",
        "|---|---|---|",
    ]
    for name, reproduced in paper["figure_20_reproduced"].items():
        lines.append(
            f"| {name} | {'yes' if reproduced else 'NO'} | {paper['nodes'][name]} |"
        )
    lines += ["", paper["note"], ""]

    lines += ["## Orderings", ""]
    for name, block in report["slices"].items():
        lines += [
            f"### {name}",
            "",
            f"{block['individuals']:,} individuals, {block['flat_cells']:,} flat "
            f"cells, {block['distinct_predicates']:,} distinct predicates",
            "",
            "| ordering | nodes | reuse | seconds |",
            "|---|---|---|---|",
        ]
        for item in block["measurements"]:
            lines.append(
                f"| {item['ordering']} | {item['nodes']:,} "
                f"| {_pct(item['reuse_rate'])} "
                f"| {block['seconds'][item['ordering']]} |"
            )
        depth = block["measurements"][0]
        lines += [
            "",
            f"Depth is identical for every ordering here — median "
            f"{depth['median_depth']}, mean {depth['mean_depth']}, p99 "
            f"{depth['p99_depth']}, max {depth['max_depth']} — because an "
            f"ordering permutes an individual's predicates without adding or "
            f"dropping any. Ordering decides how much storage is *shared*, not "
            f"how deep anyone sits.",
        ]
        first = block["h1_coverage_beats_arbitrary"]
        second = block["h2_branch_local_beats_global"]
        lines += [
            "",
            f"- **H1** coverage vs arbitrary: {first['nodes_saved']:,} nodes "
            f"({_pct(first['relative'])}) - {'holds' if first['holds'] else 'FAILS'}",
            f"- **H2** branch-local vs global: {second['nodes_saved']:,} nodes "
            f"({_pct(second['relative'])}) - {'holds' if second['holds'] else 'FAILS'}",
            f"- ordering spread, worst minus best: "
            f"{block['ordering_spread']['nodes']:,} nodes",
            "",
        ]

    scale = report["scaling"]
    third = scale["h3_depth_independent_of_vocabulary"]
    lines += [
        "## H3: access depth against vocabulary size",
        "",
        "Individuals are drawn in a fixed random order, so each row is a sample "
        "of the ontology rather than a prefix of the alphabet.",
        "",
        "| individuals | nodes | median depth | mean depth | p99 depth | max depth |",
        "|---|---|---|---|---|---|",
    ]
    for point in scale["curve"]:
        lines.append(
            f"| {point['individuals']:,} | {point['nodes']:,} "
            f"| {point['median_depth']} | {point['mean_depth']} "
            f"| {point['p99_depth']} | {point['max_depth']} |"
        )
    lines += [
        "",
        f"Vocabulary grew {scale['vocabulary_growth']}x and nodes grew "
        f"{scale['node_growth']}x, while median depth moved "
        f"{scale['median_depth_growth']}x and mean depth "
        f"{scale['mean_depth_growth']}x.",
        "",
        f"**H3 {'holds' if third['holds'] else 'FAILS'}** - {third['criterion']}.",
        "",
    ]

    fourth = report["optimality"]
    if fourth.get("samples"):
        claim = fourth["h4_greedy_is_near_optimal"]
        lines += [
            "## H4: greedy against the exhaustive optimum",
            "",
            f"{fourth['samples']} samples, predicate universe at most "
            f"{fourth['max_predicate_universe']}.",
            "",
            f"Coverage ordering hit the exact optimum on "
            f"{fourth['global_coverage_hits_optimum']} of them "
            f"({_pct(fourth['hit_rate'])}); mean excess "
            f"{fourth['mean_excess_nodes']} nodes, worst "
            f"{fourth['max_excess_nodes']}.",
            "",
            f"**H4 {'holds' if claim['holds'] else 'FAILS'}** - {claim['criterion']}.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database", type=Path, default=substrate.DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=10, help="scaling curve points")
    parser.add_argument("--samples", type=int, default=200, help="H4 sample count")
    parser.add_argument("--width", type=int, default=8,
                        help="H4 maximum predicate universe; factorial in this")
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    report = run(arguments.database, arguments.output, arguments.steps,
                 arguments.samples, arguments.width, arguments.seed)
    print(render(report))
    print(f"wrote {arguments.output / 'v683_results.json'}")


if __name__ == "__main__":
    main()
