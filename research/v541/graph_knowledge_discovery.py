#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

DEFAULT_WORKERS = 20

POSITIVE_FAMILIES = {
    "is_a": "classification",
    "instance_of": "classification",
    "defined_as": "definition",
    "has_property": "property",
    "has": "possession",
    "has_part": "composition",
    "part_of": "composition",
    "capable_of": "capability",
    "causes": "causality",
    "located_in": "location",
    "used_for": "purpose",
    "made_of": "composition",
}

NEGATIVE_RELATIONS = {
    "have_not", "has_not", "not_has", "does_not_have",
    "cannot", "not_capable_of", "does_not", "not_is_a",
}

# Conservative relation compositions. These are intentionally narrow.
SAFE_COMPOSITIONS = {
    ("is_a", "has_part"): "has_part",
    ("instance_of", "has_part"): "has_part",
    ("is_a", "has"): "has",
    ("instance_of", "has"): "has",
    ("is_a", "has_property"): "has_property",
    ("instance_of", "has_property"): "has_property",
    ("has_part", "part_of"): "has_part",
    ("has", "part_of"): "has",
}

BLOCKED_PREDICATES = {
    "in_domain", "domain", "source", "provenance", "dataset", "node_type",
    "type", "label", "nsubj", "nsubjpass", "obj", "dobj", "iobj", "ccomp",
    "xcomp", "amod", "advmod", "nmod", "obl", "oblique", "root", "dep",
    "aux", "auxpass", "cop", "det", "case", "mark", "punct", "conj", "cc",
    "compound", "appos", "acl", "advcl",
}

EXCLUDED_TABLE_WORDS = {
    "live", "conversation", "session", "turn", "history", "message", "utterance"
}

GENERIC_WORDS = {
    "entity", "thing", "object", "concept", "something", "someone", "word", "term"
}


@dataclass(frozen=True)
class KnowledgeAdapter:
    mode: str  # canonical or flat
    table: str
    concepts: str | None = None

    def relation_inventory_sql(self) -> str:
        if self.mode == "canonical":
            return f'SELECT predicate, COUNT(*) AS n FROM "{self.table}" WHERE predicate IS NOT NULL GROUP BY predicate ORDER BY n DESC'
        return f'SELECT predicate, COUNT(*) AS n FROM "{self.table}" WHERE predicate IS NOT NULL GROUP BY predicate ORDER BY n DESC'

    def subject_alias(self) -> str:
        if self.mode == "canonical":
            return "cs.canonical"
        return "f.subject"

    def object_expr(self) -> str:
        if self.mode == "canonical":
            return "COALESCE(co.canonical, f.object_text)"
        return "f.object_text"

    def base_from(self) -> str:
        if self.mode == "canonical":
            return f'''FROM "{self.table}" f
LEFT JOIN "{self.concepts}" cs ON cs.concept_id=f.subject_id
LEFT JOIN "{self.concepts}" co ON co.concept_id=f.object_id'''
        return f'FROM "{self.table}" f'

    def where_answerable(self) -> str:
        cols = CURRENT_SCHEMA[self.table]
        if "answerable" in cols:
            return "AND f.answerable=1"
        return ""

    def where_object(self) -> str:
        if self.mode == "canonical":
            return "AND (f.object_id IS NOT NULL OR f.object_text IS NOT NULL)"
        return "AND f.object_text IS NOT NULL"


CURRENT_SCHEMA: dict[str, list[str]] = {}


def connect(path: str) -> sqlite3.Connection:
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA mmap_size=536870912")
    con.execute("PRAGMA cache_size=-65536")
    return con


def inspect_schema(con: sqlite3.Connection) -> dict[str, list[str]]:
    global CURRENT_SCHEMA
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    info: dict[str, list[str]] = {}
    for table in tables:
        info[table] = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
    CURRENT_SCHEMA = info
    return info


def table_rows(con: sqlite3.Connection, table: str) -> int:
    try:
        return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except sqlite3.Error:
        return 0


def pick_adapter(con: sqlite3.Connection, info: dict[str, list[str]], forced: str | None) -> tuple[KnowledgeAdapter, list[dict]]:
    candidates = []

    for table, cols_list in info.items():
        cols = set(cols_list)
        low = table.lower()
        excluded = any(word in low for word in EXCLUDED_TABLE_WORDS)
        rows = table_rows(con, table)

        # Canonical long-term semantic graph: facts + concepts.
        if {"subject_id", "predicate", "object_id", "object_text"}.issubset(cols) and "facts" in low:
            if "concepts" in info and {"concept_id", "canonical"}.issubset(set(info["concepts"])):
                candidates.append({
                    "table": table, "mode": "canonical", "rows": rows,
                    "excluded": excluded, "score": 10_000_000 + rows,
                })
                continue

        # Generic graph edge table.
        if {"subject", "predicate", "object_text"}.issubset(cols):
            score = rows
            if "fact" in low or "edge" in low or "knowledge" in low or "semantic" in low:
                score += 1_000_000
            if excluded:
                score -= 5_000_000
            candidates.append({
                "table": table, "mode": "flat", "rows": rows,
                "excluded": excluded, "score": score,
            })

    if forced:
        if forced not in info:
            raise RuntimeError(f"Requested --table {forced!r} does not exist.")
        cols = set(info[forced])
        if {"subject_id", "predicate", "object_id", "object_text"}.issubset(cols) and "concepts" in info:
            return KnowledgeAdapter("canonical", forced, "concepts"), candidates
        if {"subject", "predicate", "object_text"}.issubset(cols):
            return KnowledgeAdapter("flat", forced, None), candidates
        raise RuntimeError(f"Table {forced!r} is not a supported knowledge-edge schema.")

    viable = [c for c in candidates if not c["excluded"] and c["rows"] > 0]
    if not viable:
        raise RuntimeError(
            "No non-live knowledge graph table found. Available tables: "
            + ", ".join(sorted(info))
        )
    chosen = max(viable, key=lambda x: x["score"])
    return KnowledgeAdapter(chosen["mode"], chosen["table"], "concepts" if chosen["mode"] == "canonical" else None), candidates


def relation_inventory(con: sqlite3.Connection, adapter: KnowledgeAdapter) -> list[dict]:
    return [
        {
            "predicate": row["predicate"],
            "count": int(row["n"]),
            "family": POSITIVE_FAMILIES.get(
                row["predicate"],
                "negative" if row["predicate"] in NEGATIVE_RELATIONS else "other",
            ),
        }
        for row in con.execute(adapter.relation_inventory_sql())
    ]


def natural_question(predicate: str, subject: str, obj: str) -> str:
    s = str(subject).strip()
    o = str(obj).strip()
    if predicate in {"is_a", "instance_of"}:
        return f"Is {s} a {o}?"
    if predicate == "defined_as":
        return f"Is {s} defined as {o}?"
    if predicate in {"has", "has_part"}:
        return f"Does {s} have {o}?"
    if predicate == "part_of":
        return f"Is {s} part of {o}?"
    if predicate == "has_property":
        return f"Is {s} {o}?"
    if predicate == "capable_of":
        return f"Can {s} {o}?"
    if predicate == "causes":
        return f"Does {s} cause {o}?"
    if predicate == "located_in":
        return f"Is {s} located in {o}?"
    if predicate == "used_for":
        return f"Is {s} used for {o}?"
    if predicate == "made_of":
        return f"Is {s} made of {o}?"
    return f"Does {s} {predicate.replace('_', ' ')} {o}?"


def fact_rows(con: sqlite3.Connection, adapter: KnowledgeAdapter, predicate_filter: list[str] | None = None, limit: int = 1000):
    params: list = []
    where = ["f.predicate IS NOT NULL", adapter.where_object()[4:] if adapter.where_object().startswith("AND ") else adapter.where_object()]
    if adapter.mode == "canonical":
        where = ["f.predicate IS NOT NULL", "(f.object_id IS NOT NULL OR f.object_text IS NOT NULL)", "cs.canonical IS NOT NULL"]
    else:
        where = ["f.predicate IS NOT NULL", "f.object_text IS NOT NULL", "f.subject IS NOT NULL"]
    if "answerable" in CURRENT_SCHEMA[adapter.table]:
        where.append("f.answerable=1")
    if predicate_filter:
        where.append("f.predicate IN (" + ",".join("?" * len(predicate_filter)) + ")")
        params.extend(predicate_filter)
    sql = f'''SELECT {adapter.subject_alias()} AS subject, f.predicate, {adapter.object_expr()} AS object_text
              {adapter.base_from()}
              WHERE {' AND '.join(where)}
                AND {adapter.object_expr()} IS NOT NULL
              LIMIT ?'''
    params.append(int(limit))
    return [dict(r) for r in con.execute(sql, params)]


def sample_direct(con: sqlite3.Connection, adapter: KnowledgeAdapter, n: int, seed: int) -> list[dict]:
    allowed = [
        r["predicate"]
        for r in relation_inventory(con, adapter)
        if r["predicate"] in POSITIVE_FAMILIES and r["predicate"] not in BLOCKED_PREDICATES
    ]
    if not allowed:
        return []
    rng = random.Random(seed)
    rng.shuffle(allowed)
    per = max(1, math.ceil(n / len(allowed)))
    out = []
    seen = set()
    for predicate in allowed:
        for row in fact_rows(con, adapter, [predicate], per):
            key = (row["subject"], row["predicate"], row["object_text"])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "question": natural_question(predicate, row["subject"], row["object_text"]),
                "status": "SUPPORTED",
                "kind": "direct",
                "subject": row["subject"],
                "predicate": predicate,
                "object": row["object_text"],
                "path": [[row["subject"], predicate, row["object_text"]]],
            })
            if len(out) >= n:
                return out
    return out[:n]


def query_edges(con: sqlite3.Connection, adapter: KnowledgeAdapter, subjects: list[str], predicates: tuple[str, ...], per_node: int) -> list[dict]:
    if not subjects:
        return []
    if adapter.mode == "flat":
        placeholders_s = ",".join("?" * len(subjects))
        placeholders_p = ",".join("?" * len(predicates))
        sql = f'''SELECT f.subject AS subject, f.predicate AS predicate, f.object_text AS object_text
                  FROM "{adapter.table}" f
                  WHERE f.subject IN ({placeholders_s})
                    AND f.predicate IN ({placeholders_p})
                    AND f.object_text IS NOT NULL
                  LIMIT ?'''
        return [dict(r) for r in con.execute(sql, [*subjects, *predicates, int(per_node * max(1, len(subjects)))])]
    # Canonical: resolve all requested subject names to IDs, then retrieve edges in one query.
    placeholders = ",".join("?" * len(subjects))
    pred_ph = ",".join("?" * len(predicates))
    sql = f'''SELECT cs.canonical AS subject, f.predicate AS predicate,
                     COALESCE(co.canonical, f.object_text) AS object_text
              FROM "{adapter.table}" f
              JOIN "{adapter.concepts}" cs ON cs.concept_id=f.subject_id
              LEFT JOIN "{adapter.concepts}" co ON co.concept_id=f.object_id
              WHERE lower(cs.canonical) IN ({placeholders})
                AND f.predicate IN ({pred_ph})
                AND COALESCE(co.canonical, f.object_text) IS NOT NULL
              LIMIT ?'''
    lowered = [s.lower() for s in subjects]
    return [dict(r) for r in con.execute(sql, [*lowered, *predicates, int(per_node * max(1, len(subjects)))])]


def mine_indirect(path: str, adapter_tuple: tuple[str, str, str | None], n: int, max_hops: int, workers: int, seed: int, per_node: int) -> list[dict]:
    adapter = KnowledgeAdapter(*adapter_tuple)
    con = connect(path)
    # Start from subjects with many answerable facts. This is discovery, not exhaustive traversal.
    cols = CURRENT_SCHEMA[adapter.table]
    if adapter.mode == "canonical":
        sql = f'''SELECT c.canonical AS subject, COUNT(*) AS n
                  FROM "{adapter.table}" f JOIN "{adapter.concepts}" c ON c.concept_id=f.subject_id
                  WHERE c.canonical IS NOT NULL {'AND f.answerable=1' if 'answerable' in cols else ''}
                  GROUP BY c.canonical ORDER BY n DESC LIMIT ?'''
    else:
        sql = f'''SELECT f.subject AS subject, COUNT(*) AS n FROM "{adapter.table}" f
                  WHERE f.subject IS NOT NULL GROUP BY f.subject ORDER BY n DESC LIMIT ?'''
    seed_count = max(500, min(20_000, n * 20))
    starts = [str(r["subject"]) for r in con.execute(sql, (seed_count,))]
    random.Random(seed).shuffle(starts)
    starts = starts[:seed_count]

    chunks = [starts[i::max(1, workers)] for i in range(max(1, workers)) if starts[i::max(1, workers)]]
    args = [(path, adapter_tuple, chunk, max_hops, per_node) for chunk in chunks]
    if workers > 1 and len(args) > 1:
        with mp.Pool(workers) as pool:
            parts = pool.map(indirect_worker, args)
    else:
        parts = [indirect_worker(args[0])] if args else []
    out = []
    seen = set()
    for part in parts:
        for item in part:
            key = (item["subject"], item["predicate"], item["object"], tuple(map(tuple, item["path"])))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= n:
                return out
    return out


def indirect_worker(args):
    path, adapter_tuple, starts, max_hops, per_node = args
    adapter = KnowledgeAdapter(*adapter_tuple)
    con = connect(path)
    predicates = tuple(POSITIVE_FAMILIES.keys())
    frontier = [(s, []) for s in starts]
    answers = []
    seen_nodes = set(starts)
    for hop in range(1, max_hops + 1):
        subject_names = list(dict.fromkeys(s for s, _ in frontier))
        edges = query_edges(con, adapter, subject_names, predicates, per_node)
        by_subject: dict[str, list[dict]] = {}
        for e in edges:
            by_subject.setdefault(str(e["subject"]).lower(), []).append(e)
        next_frontier = []
        for subject, path0 in frontier:
            for edge in by_subject.get(str(subject).lower(), []):
                np = path0 + [[edge["subject"], edge["predicate"], edge["object_text"]]]
                if len(np) >= 2:
                    composed = compose_path(np)
                    if composed:
                        q_subject, q_pred, q_obj = composed
                        answers.append({
                            "question": natural_question(q_pred, q_subject, q_obj),
                            "status": "SUPPORTED",
                            "kind": "indirect",
                            "subject": q_subject,
                            "predicate": q_pred,
                            "object": q_obj,
                            "hops": len(np),
                            "path": np,
                        })
                obj = str(edge["object_text"] or "").strip()
                if obj and len(obj) <= 120 and obj.lower() not in seen_nodes and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _'/-]{0,119}", obj):
                    seen_nodes.add(obj.lower())
                    next_frontier.append((obj, np))
        frontier = next_frontier[: max(32, per_node * 2)]
        if not frontier:
            break
        if len(answers) >= 200:
            break
    return answers


def compose_path(path: list[list[str]]) -> tuple[str, str, str] | None:
    if len(path) == 2:
        (s1, p1, o1), (s2, p2, o2) = path
        if o1.lower() != s2.lower():
            return None
        inferred = SAFE_COMPOSITIONS.get((p1, p2))
        if inferred:
            return s1, inferred, o2
    if len(path) == 3:
        p1, p2, p3 = path[0][1], path[1][1], path[2][1]
        first = SAFE_COMPOSITIONS.get((p1, p2))
        if first and SAFE_COMPOSITIONS.get((first, p3)):
            if path[0][2].lower() == path[1][0].lower() and path[1][2].lower() == path[2][0].lower():
                return path[0][0], SAFE_COMPOSITIONS[(first, p3)], path[2][2]
    return None


def discover_negative(con: sqlite3.Connection, adapter: KnowledgeAdapter, n: int) -> list[dict]:
    rels = tuple(NEGATIVE_RELATIONS)
    if adapter.mode == "canonical":
        sql = f'''SELECT cs.canonical AS subject, f.predicate, COALESCE(co.canonical,f.object_text) AS object_text
                  FROM "{adapter.table}" f
                  JOIN "{adapter.concepts}" cs ON cs.concept_id=f.subject_id
                  LEFT JOIN "{adapter.concepts}" co ON co.concept_id=f.object_id
                  WHERE f.predicate IN ({','.join('?'*len(rels))})
                    AND cs.canonical IS NOT NULL
                    AND COALESCE(co.canonical,f.object_text) IS NOT NULL
                  LIMIT ?'''
    else:
        sql = f'''SELECT subject,predicate,object_text FROM "{adapter.table}"
                  WHERE predicate IN ({','.join('?'*len(rels))}) AND subject IS NOT NULL AND object_text IS NOT NULL LIMIT ?'''
    rows = con.execute(sql, [*rels, int(n)]).fetchall()
    out = []
    for r in rows:
        s, p, o = r[0], r[1], r[2]
        out.append({
            "question": f"Does {s} have {o}?",
            "status": "REFUTED",
            "kind": "explicit_negative",
            "subject": s,
            "predicate": p,
            "object": o,
            "path": [[s, p, o]],
        })
    return out


def discover_unknown(con: sqlite3.Connection, adapter: KnowledgeAdapter, n: int, seed: int) -> list[dict]:
    relations = [p for p in POSITIVE_FAMILIES if p in {"is_a", "has_part", "has", "has_property", "capable_of", "part_of"}]
    rows = fact_rows(con, adapter, relations, max(10_000, n * 100))
    by_pred: dict[str, set[str]] = {p: set() for p in relations}
    positives = set()
    for r in rows:
        s, p, o = str(r["subject"]), str(r["predicate"]), str(r["object_text"])
        by_pred.setdefault(p, set()).add(o)
        positives.add((s.lower(), p, o.lower()))
    rng = random.Random(seed)
    out = []
    seen = set()
    for r in rows:
        s, p, o = str(r["subject"]), str(r["predicate"]), str(r["object_text"])
        pool = list(by_pred.get(p, ()))
        rng.shuffle(pool)
        for alt in pool[:50]:
            if alt.lower() == o.lower() or (s.lower(), p, alt.lower()) in positives:
                continue
            key = (s.lower(), p, alt.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "question": natural_question(p, s, alt),
                "status": "UNKNOWN",
                "kind": "hard_unknown",
                "subject": s,
                "predicate": p,
                "object": alt,
                "contrast_object": o,
            })
            break
        if len(out) >= n:
            return out
    return out


def run(args):
    started = time.perf_counter()
    con = connect(args.memory)
    info = inspect_schema(con)
    adapter, candidates = pick_adapter(con, info, args.table)

    print("=== V541 DB-NATIVE KNOWLEDGE & GOAL DISCOVERY ===")
    print(f"memory       : {args.memory}")
    print(f"knowledge    : {adapter.table} ({adapter.mode})")
    print(f"workers      : {args.workers}")
    print(f"LLM          : NOT USED")
    print(f"conversation : NOT USED")
    print()
    print("=== KNOWLEDGE TABLE CANDIDATES ===")
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        print(f"  {c['table']:<28} mode={c['mode']:<9} rows={c['rows']:<10d} excluded={c['excluded']}")
    print()

    rels = relation_inventory(con, adapter)
    print("=== ACTUAL RELATION INVENTORY ===")
    for r in rels[:60]:
        print(f"  {r['predicate']:<28} {r['count']:>10}  family={r['family']}")

    t = time.perf_counter(); direct = sample_direct(con, adapter, args.direct, args.seed)
    print(f"\n[DISCOVERY] direct supported={len(direct)} seconds={time.perf_counter()-t:.3f}")

    adapter_tuple = (adapter.mode, adapter.table, adapter.concepts)
    t = time.perf_counter(); indirect = mine_indirect(args.memory, adapter_tuple, args.indirect, args.max_hops, args.workers, args.seed, args.per_node)
    print(f"[DISCOVERY] indirect supported={len(indirect)} seconds={time.perf_counter()-t:.3f}")

    t = time.perf_counter(); neg = discover_negative(con, adapter, args.negative)
    print(f"[DISCOVERY] explicit negative={len(neg)} seconds={time.perf_counter()-t:.3f}")

    t = time.perf_counter(); unk = discover_unknown(con, adapter, args.unknown, args.seed)
    print(f"[DISCOVERY] hard unknown={len(unk)} seconds={time.perf_counter()-t:.3f}")

    fam = Counter(r["family"] for r in rels if r["family"] != "other")
    print("\n=== GOALS / QUERY FAMILIES OBSERVED IN DB ===")
    if fam:
        for k, v in fam.most_common():
            print(f"  {k:<22} predicates={v}")
    else:
        print("  <none>")

    print("\n=== DISTRIBUTION ===")
    print(f"DIRECT-SUPPORTED   : {len(direct)}")
    print(f"INDIRECT-SUPPORTED : {len(indirect)}")
    print(f"REFUTED            : {len(neg)}")
    print(f"UNKNOWN            : {len(unk)}")

    def show(title, rows, formatter):
        print(f"\n=== {title} ===")
        if not rows:
            print("  <none>")
            return
        for row in rows[:args.show]:
            print("  " + formatter(row))

    show("SAMPLE DIRECT", direct, lambda r: f"{r['question']} -> SUPPORTED via {r['path']}")
    show("SAMPLE INDIRECT", indirect, lambda r: f"{r['question']} -> SUPPORTED ({r['hops']}-hop) via {r['path']}")
    show("SAMPLE REFUTED", neg, lambda r: f"{r['question']} -> REFUTED via {r['path']}")
    show("SAMPLE UNKNOWN", unk, lambda r: f"{r['question']} -> UNKNOWN (contrast={r.get('contrast_object')})")

    report = {
        "benchmark": "v541_db_native_knowledge_goal_discovery",
        "selected_knowledge_table": {"name": adapter.table, "mode": adapter.mode},
        "table_candidates": candidates,
        "relation_inventory": rels,
        "goal_families": dict(fam),
        "direct_supported": direct,
        "indirect_supported": indirect,
        "explicit_negative": neg,
        "unknown": unk,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\nJSON written:", args.json)


def main():
    ap = argparse.ArgumentParser(description="DB-native graph knowledge and goal discovery")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--table", help="Force the knowledge table; otherwise auto-detect.")
    ap.add_argument("--direct", type=int, default=500)
    ap.add_argument("--indirect", type=int, default=500)
    ap.add_argument("--unknown", type=int, default=500)
    ap.add_argument("--negative", type=int, default=100)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--per-node", type=int, default=100)
    ap.add_argument("--seed", type=int, default=541)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--json")
    args = ap.parse_args()
    args.workers = max(1, min(64, int(args.workers)))
    args.max_hops = max(1, min(4, int(args.max_hops)))
    run(args)


if __name__ == "__main__":
    main()
