from __future__ import annotations
import argparse, json, sqlite3, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description="V676 shared-memory and provenance inspector")
    ap.add_argument("--shared-memory", required=True); ap.add_argument("--events", type=int, default=30)
    args=ap.parse_args(); db=Path(args.shared_memory).resolve(); con=sqlite3.connect(str(db),timeout=10); con.row_factory=sqlite3.Row
    print("=== V676 INSPECT ==="); print("database:",db)
    for table in ("semantic_decisions","semantic_knowledge","relation_transitions","decision_evidence","knowledge_evidence","transition_evidence","checkpoint_events","merge_events","decision_resolution"):
        try: print(f"{table}: {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:,}")
        except Exception as exc: print(f"{table}: ERROR {exc!r}")
    print("\ndecision arbitration:")
    try:
        rows=con.execute("SELECT selected,support_count,confidence,contested FROM decision_resolution ORDER BY support_count DESC LIMIT 20").fetchall()
        for r in rows: print(f"  winner={r['selected']:<18} support={r['support_count']:>5} conf={r['confidence']:.3f} contested={r['contested']}")
    except Exception as exc: print("  ERROR",repr(exc))
    print("\nknowledge provenance:")
    try:
        for r in con.execute("SELECT provenance,promotion_state,COUNT(*) n FROM semantic_knowledge GROUP BY provenance,promotion_state ORDER BY n DESC").fetchall(): print(f"  {r['provenance']:<12} {r['promotion_state']:<10} {r['n']:>8,}")
    except Exception as exc: print("  ERROR",repr(exc))
    print("\n=== RELATION COMBINATIONS ===")
    print("top transitions:")
    try:
        rows=con.execute("""
            SELECT t.previous_relation,t.next_relation,t.count,t.confidence,
                   COUNT(DISTINCT e.worker_id) sources,t.derivation_depth
            FROM relation_transitions t
            LEFT JOIN transition_evidence e ON e.key=t.key
            GROUP BY t.key
            ORDER BY t.count DESC,t.confidence DESC
            LIMIT 20
        """).fetchall()
        for r in rows:
            print(
                f"  {r['previous_relation']} -> {r['next_relation']} "
                f"count={r['count']:,} conf={r['confidence']:.3f} "
                f"sources={r['sources']} depth={r['derivation_depth']}"
            )
    except Exception as exc: print("  ERROR",repr(exc))
    print("\ncandidate compositions:")
    try:
        rows=con.execute("""
            SELECT t.previous_relation,t.next_relation,t.count,t.confidence,
                   COUNT(DISTINCT e.worker_id) sources,t.derivation_depth
            FROM relation_transitions t
            LEFT JOIN transition_evidence e ON e.key=t.key
            GROUP BY t.key
            ORDER BY t.count DESC,t.confidence DESC
            LIMIT 20
        """).fetchall()
        for r in rows:
            status="eligible" if r["sources"] >= 3 and r["confidence"] >= .85 else "candidate"
            print(
                f"  {r['previous_relation']} + {r['next_relation']} "
                f"support={r['count']:,} confidence={r['confidence']:.3f} "
                f"sources={r['sources']} depth={r['derivation_depth']} status={status}"
            )
    except Exception as exc: print("  ERROR",repr(exc))
    print("\nrelation interactions:")
    try:
        rows=con.execute("""
            SELECT relation,SUM(positive) support,AVG(confidence) confidence,
                   COUNT(*) contexts
            FROM semantic_knowledge
            WHERE kind='relation_interaction_statistics'
            GROUP BY relation
            ORDER BY support DESC,confidence DESC
            LIMIT 20
        """).fetchall()
        for r in rows:
            print(f"  {r['relation']} support={r['support']:,} conf={r['confidence']:.3f} contexts={r['contexts']:,}")
    except Exception as exc: print("  ERROR",repr(exc))
    print("\nworkers / checkpoint slots:")
    try:
        rows=con.execute("SELECT * FROM worker_status ORDER BY worker_id").fetchall(); now=time.time()
        interval=300
        for r in rows:
            wid=int(r['worker_id']); slot=int((wid*interval)/max(len(rows),1)); age=now-float(r['last_seen'])
            print(f"  {wid:02d} {r['role']:<32} pid={r['pid']} age={age:6.1f}s slot@300s={slot:3d}s batches={r['batches']} learned={r['learned']} imported={r['imported']} errors={r['errors']} last_batch={float(r['last_batch_s'] or 0):.3f}s learn/s={float(r['learned_per_s'] or 0):8.1f} syncs={r['sync_count']} sync_s={float(r['last_sync_s'] or 0):.3f}")
    except Exception as exc: print("worker status error:",repr(exc))
    print("\nrecent checkpoint events:")
    try:
        rows=con.execute("SELECT * FROM checkpoint_events ORDER BY id DESC LIMIT ?",(int(args.events),)).fetchall()
        for r in reversed(rows):
            payload=json.loads(r['payload_json']); print(f"  {r['id']} worker={r['worker_id']:02d} {r['event']:<18} dt={r['duration_s']:.4f}s exported={payload.get('exported')} imported={payload.get('imported')} merged={payload.get('merged')} conflicts={payload.get('conflicts')} target={payload.get('target_slot_s')}")
    except Exception as exc: print("event error:",repr(exc))
    print("\nrecent merge conflicts:")
    try:
        rows=con.execute("SELECT table_name,action,COUNT(*) n,SUM(conflict) c FROM merge_events GROUP BY table_name,action ORDER BY n DESC").fetchall()
        for r in rows: print(f"  {r['table_name']:<22} {r['action']:<12} rows={r['n']:,} conflicts={r['c'] or 0:,}")
    except Exception as exc: print("merge event error:",repr(exc))
    con.close()
if __name__=="__main__": main()
