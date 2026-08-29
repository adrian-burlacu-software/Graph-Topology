
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from grammar import GrammarLoader, validate_grammar
from semantic_graph import build_smoke_semantic_graph


DATASET_NAME = "BabyLM-2026-Strict-Small"


def project_root_candidates(start: Path) -> list[Path]:
    """
    Return sensible project roots, starting from the caller's current working
    directory and then the script's location.
    """
    starts = [start.resolve(), Path(__file__).resolve().parent]
    roots: list[Path] = []

    for base in starts:
        for p in (base, *base.parents):
            if p not in roots:
                roots.append(p)

    return roots


def candidate_dataset_paths(cwd: Path, explicit: Path | None) -> list[Path]:
    paths: list[Path] = []

    if explicit is not None:
        explicit = explicit.expanduser()

        # First interpret it exactly as supplied.
        paths.append(explicit if explicit.is_absolute() else cwd / explicit)

        # Also try the same path relative to likely project roots. This makes
        # commands such as ".\data\BabyLM-2026-Strict-Small" work regardless
        # of whether the loader itself lives under research/v368/....
        for root in project_root_candidates(cwd):
            p = explicit if explicit.is_absolute() else root / explicit
            if p not in paths:
                paths.append(p)

    # Automatic discovery: look for the canonical dataset under ./data and
    # ancestor/data directories.
    for root in project_root_candidates(cwd):
        p = root / "data" / DATASET_NAME
        if p not in paths:
            paths.append(p)

    return paths


def resolve_dataset(cwd: Path, explicit: Path | None) -> tuple[Path | None, list[Path]]:
    candidates = candidate_dataset_paths(cwd, explicit)

    print("[1/5] Looking for BabyLM dataset...")
    print(f"      current working directory: {cwd}")
    print(f"      canonical name:             {DATASET_NAME}")

    for p in candidates:
        try:
            exists = p.exists()
        except OSError:
            exists = False

        status = "FOUND" if exists else "missing"
        print(f"      [{status:7}] {p}")

        if exists and p.is_dir():
            return p, candidates

    return None, candidates


def print_dataset_summary(path: Path, grammar) -> None:
    files = GrammarLoader().discover(path)
    print("[4/5] Dataset loaded")
    print(f"      files discovered : {len(files)}")
    print(f"      sentences       : {grammar.sentences:,}")
    print(f"      tokens           : {grammar.tokens:,}")
    print(f"      lexicon entries  : {len(grammar.lexicon):,}")
    print(f"      grammar rules    : {len(grammar.rules):,}")


def smoke() -> dict:
    print("[1/5] No dataset path supplied; running DATA-FREE smoke test")
    loader = GrammarLoader()

    synthetic = Path(__file__).resolve().parent / "smoke_corpus.txt"
    synthetic.write_text(
        "the dog chases the cat\n"
        "the cat sees the dog\n"
        "the dog eats\n",
        encoding="utf-8",
    )

    try:
        grammar = loader.load(synthetic)
        report = validate_grammar(grammar)

        graph = build_smoke_semantic_graph()
        graph.validate()

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert grammar.sentences == 3
        assert grammar.tokens == 13
        assert report["rules"] > 0
        assert report["lexicon"] >= 5
        assert report["smoke_parses"] >= 1

        print("[2/5] Semantic graph: PASS")
        print(f"      nodes={len(graph.nodes)} edges={len(graph.edges)}")
        print("[3/5] Grammar ingestion: PASS")
        print(f"      sentences={grammar.sentences} tokens={grammar.tokens}")
        print("[4/5] Grammar validation: PASS")
        print(f"      rules={len(grammar.rules)} lexicon={len(grammar.lexicon)}")
        print(f"      parse probes={report['smoke_parses']}")
        print("[5/5] RESULT: PASS")

        return {
            "status": "PASS",
            "semantic_graph": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "relations": graph.relations(),
            },
            "grammar": report,
        }
    finally:
        synthetic.unlink(missing_ok=True)


def run_real(path: Path, limit: int | None) -> dict:
    print("[2/5] Semantic graph: constructing explicit graph...")
    graph = build_smoke_semantic_graph()
    graph.validate()
    print(f"      nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print("      semantic graph: PASS")

    print("[3/5] Scanning corpus files...")
    loader = GrammarLoader()
    files = loader.discover(path)
    print(f"      found {len(files):,} supported corpus files")

    if not files:
        raise RuntimeError(
            f"No supported corpus files found under {path}. "
            "Expected .txt/.json/.jsonl/.csv/.tsv/.md files."
        )

    print("      loading grammar representation...")
    grammar = loader.load(path, limit=limit)

    print_dataset_summary(path, grammar)

    print("[5/5] Validating loaded grammar...")
    report = validate_grammar(grammar)

    print(f"      structural validation: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"      rules={report['rules']:,}")
    print(f"      lexicon={report['lexicon']:,}")
    print(f"      sentences={report['sentences']:,}")
    print(f"      tokens={report['tokens']:,}")
    print(f"      smoke parse probes={report['smoke_parses']}")

    status = "PASS" if report["valid"] and report["rules"] >= 0 else "FAIL"
    print(f"[RESULT] {status}")

    return {
        "status": status,
        "dataset": str(path.resolve()),
        "semantic_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "relations": graph.relations(),
        },
        "grammar": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load semantic graph + local BabyLM corpus with verbose status."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=(
            f"BabyLM path. From project root, this can simply be "
            f".\\data\\{DATASET_NAME}. If omitted, automatic discovery is attempted."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load the first N sentences/files worth of records.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the data-free smoke test explicitly.",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()

    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return

    path, candidates = resolve_dataset(cwd, args.path)

    if path is None:
        print("")
        print("RESULT: DATASET NOT FOUND")
        print("")
        print("What to try from the PROJECT ROOT:")
        print(
            f"  python .\\research\\v368\\load_grammar.py "
            f".\\data\\{DATASET_NAME}"
        )
        print("")
        print("Or let the loader auto-discover it:")
        print(
            "  python .\\research\\v368\\load_grammar.py"
        )
        print("")
        print("Data-free smoke test:")
        print(
            "  python .\\research\\v368\\load_grammar.py --smoke"
        )
        raise SystemExit(2)

    try:
        result = run_real(path, args.limit)
    except Exception as exc:
        print("")
        print("RESULT: LOAD FAILED")
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(3)

    print("")
    print("JSON SUMMARY")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
