from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shared-memory", required=True)
    ap.add_argument("--events", type=int, default=20)
    args=ap.parse_args()
    db=Path(args.shared_memory).resolve()
    con=sqlite3.connect(str(db), timeout=5.0)
    con.row_factory=sqlite3.Row
    print("=== V671 INSPECT ===")
    print("database:", db)
    for table in ("semantic_decisions","semantic_knowledge","relation_transitions","checkpoint_events"):
        try:
            print(f"{table}: {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:,}")
        except Exception as exc:
            print(f"{table}: ERROR {exc!r}")
    print("\nworkers:")
    try:
        rows=con.execute("SELECT * FROM worker_status ORDER BY worker_id").fetchall()
        now=time.time()
        for r in rows:
            age=now-float(r["last_seen"])
            print(f"  {int(r['worker_id']):02d} {r['role']:<32} pid={r['pid']} age={age:6.1f}s batches={r['batches']} learned={r['learned']} imported={r['imported']} errors={r['errors']}")
            if r["last_error"]:
                print("       last_error:", r["last_error"])
    except Exception as exc:
        print("worker status error:", repr(exc))
    print("\nrecent events:")
    try:
        rows=con.execute("SELECT * FROM checkpoint_events ORDER BY id DESC LIMIT ?", (int(args.events),)).fetchall()
        for r in reversed(rows):
            payload=json.loads(r["payload_json"])
            print(f"  {r['id']} worker={r['worker_id']:02d} event={r['event']} dt={r['duration_s']:.4f}s payload={payload}")
    except Exception as exc:
        print("event error:", repr(exc))
    con.close()


if __name__ == "__main__":
    main()
