from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def read_json_objects(path: Path):
    decoder = json.JSONDecoder()
    content = path.read_text(encoding="utf-8", errors="replace")
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        try:
            row, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            next_line = content.find("\n", position)
            position = len(content) if next_line < 0 else next_line + 1
            continue
        if isinstance(row, dict):
            yield row


def worker_summary(log_dir: Path):
    workers = []
    totals = Counter()
    for path in sorted(log_dir.glob("worker_*.jsonl")):
        rows = list(read_json_objects(path))
        starts = [row for row in rows if row.get("event") == "worker_start"]
        stops = [row for row in rows if row.get("event") == "worker_stop"]
        batches = [row for row in rows if row.get("event") == "analysis_batch"]
        syncs = [
            row for row in rows
            if row.get("event") in {"checkpoint_sync", "final_checkpoint"}
        ]
        errors = [row for row in rows if row.get("event") == "error"]
        latest = stops[-1] if stops else (batches[-1] if batches else {})
        sync_totals = Counter()
        for row in syncs:
            for key in ("exported", "imported", "merged", "conflicts"):
                sync_totals[key] += int(row.get(key, 0) or 0)
        worker = {
            "worker_id": latest.get("worker_id", path.stem.removeprefix("worker_")),
            "role": latest.get("role", starts[-1].get("role") if starts else None),
            "started": bool(starts),
            "stopped": bool(stops),
            "batches": int(latest.get("batches", len(batches)) or 0),
            "items": int(latest.get("items", 0) or 0),
            "learned": int(latest.get("learned", 0) or 0),
            "errors": len(errors),
            "syncs": len(syncs),
            "cpu_seconds": float(latest.get("cpu_seconds", 0) or 0),
            "cpu_utilization": float(latest.get("cpu_utilization", 0) or 0),
            **dict(sync_totals),
        }
        workers.append(worker)
        for key in ("batches", "items", "learned", "errors", "syncs", "cpu_seconds",
                    "exported", "imported", "merged", "conflicts"):
            totals[key] += worker.get(key, 0)
    return workers, dict(totals)


def query_rows(connection, query, limit=12):
    return [dict(row) for row in connection.execute(query, (limit,)).fetchall()]


def shared_summary(path: Path):
    if not path.exists():
        return {"available": False}
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        counts = {}
        for table in (
            "semantic_decisions", "semantic_knowledge", "relation_transitions",
            "decision_evidence", "knowledge_evidence", "transition_evidence",
            "checkpoint_events", "merge_events", "decision_resolution",
        ):
            counts[table] = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        return {
            "available": True,
            "counts": counts,
            "merge_actions": query_rows(
                connection,
                """SELECT table_name,action,COUNT(*) AS events,SUM(conflict) AS conflicts
                   FROM merge_events GROUP BY table_name,action
                   ORDER BY events DESC LIMIT ?""",
            ),
            "arbitration": query_rows(
                connection,
                """SELECT selected,support_count,confidence,contested
                   FROM decision_resolution
                   ORDER BY contested DESC,support_count DESC LIMIT ?""",
            ),
            "top_transitions": query_rows(
                connection,
                """SELECT t.previous_relation,t.next_relation,t.count,t.confidence,
                          COUNT(DISTINCT e.worker_id) AS sources,t.derivation_depth
                   FROM relation_transitions t
                   LEFT JOIN transition_evidence e ON e.key=t.key
                   GROUP BY t.key ORDER BY t.count DESC,t.confidence DESC LIMIT ?""",
            ),
            "top_interactions": query_rows(
                connection,
                """SELECT relation,SUM(positive) AS support,AVG(confidence) AS confidence,
                          COUNT(*) AS contexts
                   FROM semantic_knowledge
                   WHERE kind='relation_interaction_statistics'
                   GROUP BY relation ORDER BY support DESC,confidence DESC LIMIT ?""",
            ),
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Write a compact V678 overnight-run worker/checkpoint summary."
    )
    parser.add_argument("--worker-log-dir", default="./results/v678_workers")
    parser.add_argument("--shared-memory", default="./results/v678_shared_memory.sqlite")
    parser.add_argument("--output", default="./results/v678/worker_summary.jsonl")
    args = parser.parse_args()

    workers, totals = worker_summary(Path(args.worker_log_dir))
    shared = shared_summary(Path(args.shared_memory))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"record_type": "run_summary", "worker_count": len(workers), "totals": totals},
        {"record_type": "workers", "workers": workers},
        {"record_type": "shared_checkpoint", **shared},
    ]
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} summary records to {output.resolve()}")


if __name__ == "__main__":
    main()
