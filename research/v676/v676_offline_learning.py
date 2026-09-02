from __future__ import annotations

import argparse
import hashlib
import os
import random
import sqlite3
import time
import json
from pathlib import Path

from v676_memory import RamSemanticMemory, SharedCheckpoint

RELATION_LANES = [
    "antonym_structure",
    "synonym_structure",
    "hypernym_structure",
    "hyponym_structure",
    "meronym_structure",
    "holonym_structure",
    "property_structure",
    "capability_structure",
    "cause_structure",
    "purpose_structure",
    "location_structure",
    "association_structure",
    "contrast_structure",
    "relation_inverse",
    "relation_symmetry",
    "counterrelation_mining",
    "relation_composition",
    "relation_interaction_statistics",
    "graph_health_sampling",
]

RELATION_GROUPS = {
    "antonym_structure": ["antonym"],
    "synonym_structure": ["synonym", "similar_to"],
    "hypernym_structure": ["is_a"],
    "hyponym_structure": ["is_a"],
    "meronym_structure": ["has_part"],
    "holonym_structure": ["part_of", "has_a"],
    "property_structure": ["has_property"],
    "capability_structure": ["capable_of"],
    "cause_structure": ["causes"],
    "purpose_structure": ["used_for"],
    "location_structure": ["at_location"],
    "association_structure": ["related_to"],
    "contrast_structure": ["antonym"],
    "relation_inverse": [],
    "relation_symmetry": [],
    "counterrelation_mining": [],
    "relation_composition": [],
    "relation_interaction_statistics": [],
    "graph_health_sampling": [],
}


def read_counts(conn):
    return {
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
    }


def sample_subjects(conn, seed: int, worker_id: int, batch_size: int = 64):
    # Deterministic sampling by rowid-ish hash without scanning the whole graph.
    # Each worker starts from a different offset and walks lexical nodes.
    total = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE node_type IN ('concept','synset')"
    ).fetchone()[0]
    if not total:
        return []
    offset = (seed + worker_id * 7919) % int(total)
    rows = conn.execute(
        """
        SELECT node
        FROM nodes
        WHERE node_type IN ('concept','synset')
        ORDER BY node
        LIMIT ? OFFSET ?
        """,
        (int(batch_size), int(offset)),
    ).fetchall()
    return [str(r[0]) for r in rows]


def fetch_outgoing(conn, node, limit=64):
    return conn.execute(
        "SELECT subject,relation,object FROM edges WHERE subject=? LIMIT ?",
        (node, int(limit)),
    ).fetchall()


def fetch_incoming(conn, node, limit=64):
    return conn.execute(
        "SELECT subject,relation,object FROM edges WHERE object=? LIMIT ?",
        (node, int(limit)),
    ).fetchall()


def feature_for_pair(conn, subject, relation, obj):
    out_subject = {str(r[1]) for r in fetch_outgoing(conn, subject, 64)}
    out_object = {str(r[1]) for r in fetch_outgoing(conn, obj, 64)}
    in_subject = {str(r[1]) for r in fetch_incoming(conn, subject, 64)}
    in_object = {str(r[1]) for r in fetch_incoming(conn, obj, 64)}
    shared_out = len(out_subject & out_object)
    shared_in = len(in_subject & in_object)
    return {
        "subject_out_degree": len(out_subject),
        "object_out_degree": len(out_object),
        "subject_in_degree": len(in_subject),
        "object_in_degree": len(in_object),
        "shared_out_relations": shared_out,
        "shared_in_relations": shared_in,
        "relation": relation,
    }


def run_lane(conn, ram, lane, seed, worker_id, batch_no):
    rng = random.Random(seed + worker_id * 100003 + batch_no * 9973)
    subjects = sample_subjects(conn, seed + batch_no * 17, worker_id, 64)
    learned = 0
    inspected = 0

    if lane in RELATION_GROUPS and RELATION_GROUPS[lane]:
        relations = RELATION_GROUPS[lane]
        for subject in subjects:
            rows = fetch_outgoing(conn, subject, 96)
            for row in rows:
                relation = str(row[1])
                if relation not in relations:
                    continue
                obj = str(row[2])
                feature = feature_for_pair(conn, subject, relation, obj)
                feature["lane"] = lane
                ram.upsert_knowledge(lane, subject, relation, obj, feature, positive=1, confidence=0.5, source="offline_graph_analysis")
                learned += 1
                inspected += 1
                if len(rows) > 20 and rng.random() < 0.08:
                    break

    elif lane == "relation_inverse":
        for subject in subjects:
            rows = fetch_outgoing(conn, subject, 64)
            for row in rows:
                rel, obj = str(row[1]), str(row[2])
                back = conn.execute("SELECT 1 FROM edges WHERE subject=? AND relation=? AND object=? LIMIT 1", (obj, rel, subject)).fetchone()
                if back:
                    ram.upsert_knowledge(lane, subject, rel, obj, {"inverse_same_relation": True}, positive=1, confidence=0.9, source="offline_graph_analysis", provenance="derived", derivation_depth=1)
                inspected += 1

    elif lane == "relation_symmetry":
        for subject in subjects:
            rows = fetch_outgoing(conn, subject, 64)
            for row in rows:
                rel, obj = str(row[1]), str(row[2])
                back = conn.execute("SELECT 1 FROM edges WHERE subject=? AND relation=? AND object=? LIMIT 1", (obj, rel, subject)).fetchone()
                if back:
                    ram.upsert_knowledge(lane, subject, rel, obj, {"symmetric": True}, positive=1, confidence=0.8, source="offline_graph_analysis", provenance="derived", derivation_depth=1)
                inspected += 1

    elif lane == "counterrelation_mining":
        # Mine hard negatives: semantically close neighboring concepts connected
        # by a different relation. The point is to learn what a relation is not.
        for subject in subjects:
            rows = fetch_outgoing(conn, subject, 80)
            if not rows:
                continue
            first = rows[:8]
            for pos in first:
                pos_rel, pos_obj = str(pos[1]), str(pos[2])
                for neg in rows[8:20]:
                    neg_rel, neg_obj = str(neg[1]), str(neg[2])
                    if neg_rel == pos_rel or neg_obj == pos_obj:
                        continue
                    ram.upsert_knowledge(lane, subject, pos_rel, neg_obj, {"positive_relation": pos_rel, "distractor_relation": neg_rel, "positive_object": pos_obj}, negative=1, confidence=0.5, source="offline_counterrelation", provenance="derived", derivation_depth=1)
                    learned += 1
                    inspected += 1
                    break

    elif lane == "relation_composition":
        # V671 generated an uncontrolled Cartesian product. V676 deliberately
        # caps both fanout and total derived records per batch. Composition is
        # useful training evidence, but it is NOT graph truth.
        fanout = max(1, int(getattr(ram, "composition_fanout", 4)))
        max_derived = max(1, int(getattr(ram, "composition_max", 2000)))
        produced = 0
        for subject in subjects:
            if ram.composition_produced >= max_derived:
                break
            rows = fetch_outgoing(conn, subject, 24)
            for row in rows[:fanout]:
                if ram.composition_produced >= max_derived:
                    break
                mid, rel1 = str(row[2]), str(row[1])
                for row2 in fetch_outgoing(conn, mid, 12)[:fanout]:
                    if ram.composition_produced >= max_derived:
                        break
                    rel2, obj = str(row2[1]), str(row2[2])
                    ram.upsert_transition(rel1, rel2, confidence=0.5, count=1, provenance="derived", derivation_depth=2)
                    ram.upsert_knowledge(lane, subject, rel1 + "->" + rel2, obj, {"middle": mid, "depth": 2}, positive=1, confidence=0.4, source="offline_composition", provenance="derived", derivation_depth=2)
                    learned += 1; inspected += 1; produced += 1
                    ram.composition_produced += 1

    elif lane == "relation_interaction_statistics":
        # Relations sharing a subject are interaction evidence, not a path
        # composition. Store it separately so composition statistics remain
        # interpretable and the controller can later learn contextual pairings.
        for subject in subjects:
            relations = sorted({str(row[1]) for row in fetch_outgoing(conn, subject, 64)})
            for index, first in enumerate(relations[:12]):
                for second in relations[index + 1:12]:
                    ram.upsert_knowledge(
                        lane,
                        subject,
                        f"{first}+{second}",
                        None,
                        {"relations": [first, second], "interaction": "co_occurs"},
                        positive=1,
                        confidence=0.5,
                        source="offline_relation_interaction",
                        provenance="derived",
                        derivation_depth=1,
                    )
                    learned += 1
                    inspected += 1

    elif lane == "graph_health_sampling":
        for subject in subjects:
            rows = fetch_outgoing(conn, subject, 32)
            rels = [str(r[1]) for r in rows]
            uniq = len(set(rels))
            ram.upsert_knowledge(lane, subject, "degree_profile", None, {"edges": len(rows), "unique_relations": uniq}, positive=1, confidence=0.5, source="offline_graph_health")
            inspected += 1

    return inspected, learned


def worker_main(args, worker_id: int, stop_event=None):
    role = f"offline:{RELATION_LANES[worker_id % len(RELATION_LANES)]}"
    ram = RamSemanticMemory(worker_id)
    ram.composition_fanout = int(getattr(args, "composition_fanout", 4))
    ram.composition_max = int(getattr(args, "composition_max", 2000))
    shared = SharedCheckpoint(args.shared_memory, worker_id, args.total_workers, args.checkpoint_seconds)
    log_path = Path(args.worker_log_dir) / f"worker_{worker_id:02d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    batches = items = learned_total = imported_total = errors = 0
    elapsed = 0.0
    learned = 0

    def log(event, **payload):
        row = {"timestamp": time.time(), "worker_id": worker_id, "role": role, "event": event, **payload}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()

    log("worker_start", pid=os.getpid())
    shared.heartbeat(role, os.getpid())
    conn = sqlite3.connect(f"file:{Path(args.database).resolve()}?mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if args.duration_seconds and time.time() - started >= args.duration_seconds:
                break
            batch_started = time.perf_counter()
            lane = RELATION_LANES[worker_id % len(RELATION_LANES)]
            try:
                inspected, learned = run_lane(conn, ram, lane, args.seed, worker_id, batches)
                batches += 1
                items += inspected
                learned_total += learned
                elapsed = time.perf_counter() - batch_started
                log("analysis_batch", batch=batches, lane=lane, inspected=inspected, learned=learned, duration_s=elapsed, inspected_per_s=(inspected/max(elapsed,1e-9)), learned_per_s=(learned/max(elapsed,1e-9)), ram=ram.counts(), provenance=ram.provenance_counts())
            except Exception as exc:
                errors += 1
                shared.heartbeat(role, os.getpid(), batches=batches, items=items, learned=learned_total, imported=imported_total, errors=errors, last_error=repr(exc))
                log("error", stage="analysis_batch", error=repr(exc))
                time.sleep(1.0)

            # Evenly-spaced interval slots provide staggered write slots. We still check
            # frequently, but only the worker owning the current slot writes.
            if shared.should_sync():
                try:
                    shared.heartbeat(role, os.getpid(), batches=batches, items=items, learned=learned_total, imported=imported_total, errors=errors, last_batch_s=elapsed, learned_per_s=(learned/max(elapsed,1e-9)))
                    sync = shared.sync(ram, role, os.getpid())
                    imported_total += sync.get("imported", 0)
                    log("checkpoint_sync", **sync)
                except Exception as exc:
                    errors += 1
                    log("error", stage="checkpoint_sync", error=repr(exc))

            sleep_s = max(0.05, float(args.batch_sleep))
            if stop_event is not None:
                stop_event.wait(sleep_s)
            else:
                time.sleep(sleep_s)

        # Best-effort final merge when requested to stop.
        try:
            sync = shared.sync(ram, role, os.getpid(), force=True)
            log("final_checkpoint", **sync)
        except Exception as exc:
            log("error", stage="final_checkpoint", error=repr(exc))
    finally:
        conn.close()
        shared.heartbeat(role, os.getpid(), batches=batches, items=items, learned=learned_total, imported=imported_total, errors=errors, last_batch_s=elapsed, learned_per_s=(learned/max(elapsed,1e-9)))
        shared.close()
        log("worker_stop", batches=batches, items=items, learned=learned_total, imported=imported_total, errors=errors)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", required=True)
    ap.add_argument("--shared-memory", required=True)
    ap.add_argument("--worker-log-dir", required=True)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--total-workers", type=int, default=20)
    ap.add_argument("--checkpoint-seconds", type=int, choices=(60, 300), default=300)
    ap.add_argument("--seed", type=int, default=67100)
    ap.add_argument("--batch-sleep", type=float, default=0.20)
    ap.add_argument("--duration-seconds", type=int, default=0)
    ap.add_argument("--composition-fanout", type=int, default=4)
    ap.add_argument("--composition-max", type=int, default=2000,
                    help="Maximum derived compositions per worker run")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    worker_main(args, args.worker_id)
