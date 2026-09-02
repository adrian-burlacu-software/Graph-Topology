from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _stable_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RamSemanticMemory:
    """Per-process semantic working memory backed by an in-RAM SQLite DB."""

    def __init__(self, worker_id: int):
        self.worker_id = int(worker_id)
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.executescript(
            """
            PRAGMA journal_mode=MEMORY;
            PRAGMA synchronous=OFF;

            CREATE TABLE semantic_decisions(
                key TEXT PRIMARY KEY,
                decision_type TEXT NOT NULL,
                surface TEXT NOT NULL,
                context_key TEXT NOT NULL,
                candidate_set TEXT NOT NULL,
                selected TEXT NOT NULL,
                count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                first_seen REAL NOT NULL,
                updated_unix REAL NOT NULL,
                worker_id INTEGER NOT NULL
            );

            CREATE TABLE semantic_knowledge(
                key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                subject TEXT,
                relation TEXT,
                object TEXT,
                feature_json TEXT NOT NULL,
                positive INTEGER NOT NULL DEFAULT 0,
                negative INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL,
                updated_unix REAL NOT NULL,
                worker_id INTEGER NOT NULL
            );

            CREATE TABLE relation_transitions(
                key TEXT PRIMARY KEY,
                previous_relation TEXT NOT NULL,
                next_relation TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.0,
                updated_unix REAL NOT NULL,
                worker_id INTEGER NOT NULL
            );

            CREATE TABLE worker_local_meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {
                "decisions": self.conn.execute("SELECT COUNT(*) FROM semantic_decisions").fetchone()[0],
                "knowledge": self.conn.execute("SELECT COUNT(*) FROM semantic_knowledge").fetchone()[0],
                "transitions": self.conn.execute("SELECT COUNT(*) FROM relation_transitions").fetchone()[0],
            }

    @staticmethod
    def _candidate_json(candidates: list[str]) -> str:
        return json.dumps(sorted({str(x) for x in candidates}), ensure_ascii=False, sort_keys=True)

    def lookup(self, decision_type, surface, context_text, candidates):
        if not candidates:
            return None
        context_key = _stable_key(decision_type, context_text)
        candidate_json = self._candidate_json(list(candidates))
        with self.lock:
            row = self.conn.execute(
                """
                SELECT selected,count,confidence,source,candidate_set
                FROM semantic_decisions
                WHERE decision_type=? AND context_key=? AND candidate_set=?
                ORDER BY count DESC, confidence DESC, updated_unix DESC
                LIMIT 1
                """,
                (str(decision_type), context_key, candidate_json),
            ).fetchone()
        if not row or str(row["selected"]) not in {str(x) for x in candidates}:
            return None
        return {
            "selected": str(row["selected"]),
            "count": int(row["count"]),
            "confidence": float(row["confidence"]),
            "source": str(row["source"]),
        }

    def learn(self, decision_type, surface, context_text, candidates, selected, confidence, source="online"):
        if not candidates or selected not in candidates:
            return False
        now = time.time()
        context_key = _stable_key(decision_type, context_text)
        candidate_json = self._candidate_json(list(candidates))
        key = _stable_key("decision", decision_type, surface, context_key, candidate_json, selected)
        confidence = max(0.0, min(1.0, float(confidence)))
        with self.lock:
            row = self.conn.execute(
                "SELECT count,confidence FROM semantic_decisions WHERE key=?",
                (key,),
            ).fetchone()
            if row:
                count = int(row["count"]) + 1
                avg = ((float(row["confidence"]) * (count - 1)) + confidence) / count
                self.conn.execute(
                    "UPDATE semantic_decisions SET count=?,confidence=?,updated_unix=?,source=? WHERE key=?",
                    (count, avg, now, source, key),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO semantic_decisions
                    (key,decision_type,surface,context_key,candidate_set,selected,count,confidence,source,first_seen,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (key, str(decision_type), str(surface), context_key, candidate_json, str(selected), 1, confidence, source, now, now, self.worker_id),
                )
            self.conn.commit()
        return True

    def goal_lookup(self, frame_key, candidates, min_confidence=0.85, min_count=2):
        result = self.lookup("semantic_goal", frame_key, frame_key, candidates)
        if not result:
            return None
        if result["count"] < int(min_count) or result["confidence"] < float(min_confidence):
            return None
        result["source"] = "ram_semantic_memory"
        return result

    def goal_learn(self, frame_key, candidates, selected, confidence):
        return self.learn("semantic_goal", frame_key, frame_key, candidates, selected, confidence, source="online_goal")

    def upsert_knowledge(
        self,
        kind: str,
        subject: str | None,
        relation: str | None,
        obj: str | None,
        feature: dict[str, Any],
        positive: int = 0,
        negative: int = 0,
        confidence: float = 0.0,
        source: str = "offline",
    ) -> str:
        now = time.time()
        feature_json = json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = _stable_key("knowledge", kind, subject, relation, obj, feature_json)
        with self.lock:
            row = self.conn.execute(
                "SELECT positive,negative,confidence FROM semantic_knowledge WHERE key=?",
                (key,),
            ).fetchone()
            if row:
                p = int(row["positive"]) + int(positive)
                n = int(row["negative"]) + int(negative)
                c0 = float(row["confidence"])
                total = p + n
                c = ((c0 * max(total - 1, 0)) + float(confidence)) / max(total, 1)
                self.conn.execute(
                    """UPDATE semantic_knowledge SET positive=?,negative=?,confidence=?,updated_unix=?,source=? WHERE key=?""",
                    (p, n, max(0.0, min(1.0, c)), now, source, key),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO semantic_knowledge
                    (key,kind,subject,relation,object,feature_json,positive,negative,confidence,source,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (key, kind, subject, relation, obj, feature_json, int(positive), int(negative), max(0.0, min(1.0, float(confidence))), source, now, self.worker_id),
                )
            self.conn.commit()
        return key

    def upsert_transition(self, previous_relation: str, next_relation: str, confidence: float = 0.0, count: int = 1):
        key = _stable_key("transition", previous_relation, next_relation)
        now = time.time()
        with self.lock:
            row = self.conn.execute(
                "SELECT count,confidence FROM relation_transitions WHERE key=?",
                (key,),
            ).fetchone()
            if row:
                n = int(row["count"]) + int(count)
                avg = ((float(row["confidence"]) * max(n - count, 0)) + float(confidence) * count) / max(n, 1)
                self.conn.execute("UPDATE relation_transitions SET count=?,confidence=?,updated_unix=? WHERE key=?", (n, avg, now, key))
            else:
                self.conn.execute("INSERT INTO relation_transitions VALUES(?,?,?,?,?,?,?)", (key, str(previous_relation), str(next_relation), int(count), float(confidence), now, self.worker_id))
            self.conn.commit()
        return key

    def export_records(self, limit: int = 5000, since: float = 0.0) -> dict[str, list[dict[str, Any]]]:
        with self.lock:
            decisions = [dict(r) for r in self.conn.execute("SELECT * FROM semantic_decisions WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT ?", (float(since), int(limit))).fetchall()]
            knowledge = [dict(r) for r in self.conn.execute("SELECT * FROM semantic_knowledge WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT ?", (float(since), int(limit))).fetchall()]
            transitions = [dict(r) for r in self.conn.execute("SELECT * FROM relation_transitions WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT ?", (float(since), int(limit))).fetchall()]
        return {"decisions": decisions, "knowledge": knowledge, "transitions": transitions}

    def import_records(self, payload: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
        imported = {"decisions": 0, "knowledge": 0, "transitions": 0}
        with self.lock:
            for row in payload.get("decisions", []):
                existing = self.conn.execute("SELECT count,confidence FROM semantic_decisions WHERE key=?", (row["key"],)).fetchone()
                if existing:
                    current_count = int(existing["count"])
                    incoming_count = int(row["count"])
                    if incoming_count > current_count or float(row["confidence"]) > float(existing["confidence"]):
                        self.conn.execute("""UPDATE semantic_decisions SET count=?,confidence=?,source=?,updated_unix=? WHERE key=?""", (max(current_count, incoming_count), max(float(existing["confidence"]), float(row["confidence"])), str(row.get("source", "shared")), max(float(row.get("updated_unix", 0)), time.time()), row["key"]))
                else:
                    self.conn.execute("""INSERT OR IGNORE INTO semantic_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", tuple(row.get(k) for k in ["key","decision_type","surface","context_key","candidate_set","selected","count","confidence","source","first_seen","updated_unix","worker_id"]))
                imported["decisions"] += 1
            for row in payload.get("knowledge", []):
                self.conn.execute(
                    """
                    INSERT INTO semantic_knowledge(key,kind,subject,relation,object,feature_json,positive,negative,confidence,source,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                        positive=MAX(semantic_knowledge.positive,excluded.positive),
                        negative=MAX(semantic_knowledge.negative,excluded.negative),
                        confidence=MAX(semantic_knowledge.confidence,excluded.confidence),
                        updated_unix=MAX(semantic_knowledge.updated_unix,excluded.updated_unix)
                    """,
                    tuple(row.get(k) for k in ["key","kind","subject","relation","object","feature_json","positive","negative","confidence","source","updated_unix","worker_id"]),
                )
                imported["knowledge"] += 1
            for row in payload.get("transitions", []):
                self.conn.execute(
                    """
                    INSERT INTO relation_transitions(key,previous_relation,next_relation,count,confidence,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                        count=MAX(relation_transitions.count,excluded.count),
                        confidence=MAX(relation_transitions.confidence,excluded.confidence),
                        updated_unix=MAX(relation_transitions.updated_unix,excluded.updated_unix)
                    """,
                    tuple(row.get(k) for k in ["key","previous_relation","next_relation","count","confidence","updated_unix","worker_id"]),
                )
                imported["transitions"] += 1
            self.conn.commit()
        return imported

    def relation_prior(self, previous_relations: list[str], candidate_relation: str) -> float:
        if not previous_relations:
            return 0.0
        with self.lock:
            vals=[]
            for prev in previous_relations[-3:]:
                row=self.conn.execute("SELECT count,confidence FROM relation_transitions WHERE previous_relation=? AND next_relation=?", (str(prev), str(candidate_relation))).fetchone()
                if row:
                    vals.append(min(1.0, float(row["confidence"])) * min(1.0, float(row["count"])/10.0))
        return max(vals, default=0.0)


class SharedCheckpoint:
    """Serialized checkpoint store. Each process owns a RAM DB and occasionally merges here."""

    def __init__(self, path: str | Path, worker_id: int, total_workers: int, interval_seconds: int = 300):
        self.path = Path(path)
        self.worker_id = int(worker_id)
        self.total_workers = int(total_workers)
        self.interval_seconds = int(interval_seconds)
        if self.interval_seconds < max(total_workers + 1, 21):
            raise ValueError("checkpoint interval must leave at least one second of staggered slots per worker")
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=5.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS semantic_decisions(
                key TEXT PRIMARY KEY, decision_type TEXT NOT NULL, surface TEXT NOT NULL,
                context_key TEXT NOT NULL, candidate_set TEXT NOT NULL, selected TEXT NOT NULL,
                count INTEGER NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL,
                first_seen REAL NOT NULL, updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_knowledge(
                key TEXT PRIMARY KEY, kind TEXT NOT NULL, subject TEXT, relation TEXT, object TEXT,
                feature_json TEXT NOT NULL, positive INTEGER NOT NULL, negative INTEGER NOT NULL,
                confidence REAL NOT NULL, source TEXT NOT NULL, updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relation_transitions(
                key TEXT PRIMARY KEY, previous_relation TEXT NOT NULL, next_relation TEXT NOT NULL,
                count INTEGER NOT NULL, confidence REAL NOT NULL, updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS worker_status(
                worker_id INTEGER PRIMARY KEY, role TEXT NOT NULL, pid INTEGER, last_seen REAL NOT NULL,
                last_sync REAL, batches INTEGER NOT NULL DEFAULT 0, items INTEGER NOT NULL DEFAULT 0,
                learned INTEGER NOT NULL DEFAULT 0, imported INTEGER NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            );
            CREATE TABLE IF NOT EXISTS checkpoint_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id INTEGER NOT NULL, event TEXT NOT NULL,
                ts REAL NOT NULL, duration_s REAL NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoint_events_ts ON checkpoint_events(ts);
            """
        )
        self.conn.commit()
        self.last_bucket = None
        self.last_export_unix = 0.0
        self.last_import_unix = 0.0

    def should_sync(self, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        second = int(now) % self.interval_seconds
        # Spread 20 workers across 20 seconds using current-time modulus.
        slot = self.worker_id % min(self.interval_seconds, 20)
        bucket = int(now) // self.interval_seconds
        if second != slot or bucket == self.last_bucket:
            return False
        self.last_bucket = bucket
        return True

    def heartbeat(self, role: str, pid: int, **updates):
        with self.lock:
            row = self.conn.execute("SELECT worker_id FROM worker_status WHERE worker_id=?", (self.worker_id,)).fetchone()
            previous = self.conn.execute("SELECT * FROM worker_status WHERE worker_id=?", (self.worker_id,)).fetchone()
            values = {
                "role": role, "pid": int(pid), "last_seen": time.time(),
                "last_sync": updates.get("last_sync", previous["last_sync"] if previous else None),
                "batches": int(updates.get("batches", previous["batches"] if previous else 0)),
                "items": int(updates.get("items", previous["items"] if previous else 0)),
                "learned": int(updates.get("learned", previous["learned"] if previous else 0)),
                "imported": int(updates.get("imported", previous["imported"] if previous else 0)),
                "errors": int(updates.get("errors", previous["errors"] if previous else 0)),
                "last_error": updates.get("last_error", previous["last_error"] if previous else None),
            }
            if row:
                self.conn.execute("""UPDATE worker_status SET role=?,pid=?,last_seen=?,last_sync=COALESCE(?,last_sync),batches=?,items=?,learned=?,imported=?,errors=?,last_error=? WHERE worker_id=?""", (*values.values(), self.worker_id))
            else:
                self.conn.execute("""INSERT INTO worker_status(worker_id,role,pid,last_seen,last_sync,batches,items,learned,imported,errors,last_error) VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (self.worker_id, *values.values()))
            self.conn.commit()

    def _merge_decisions(self, rows):
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO semantic_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  count=MAX(semantic_decisions.count,excluded.count),
                  confidence=MAX(semantic_decisions.confidence,excluded.confidence),
                  updated_unix=MAX(semantic_decisions.updated_unix,excluded.updated_unix)
                """,
                tuple(row.get(k) for k in ["key","decision_type","surface","context_key","candidate_set","selected","count","confidence","source","first_seen","updated_unix","worker_id"]),
            )

    def _merge_knowledge(self, rows):
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO semantic_knowledge VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  positive=MAX(semantic_knowledge.positive,excluded.positive),
                  negative=MAX(semantic_knowledge.negative,excluded.negative),
                  confidence=MAX(semantic_knowledge.confidence,excluded.confidence),
                  updated_unix=MAX(semantic_knowledge.updated_unix,excluded.updated_unix)
                """,
                tuple(row.get(k) for k in ["key","kind","subject","relation","object","feature_json","positive","negative","confidence","source","updated_unix","worker_id"]),
            )

    def _merge_transitions(self, rows):
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO relation_transitions VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                  count=MAX(relation_transitions.count,excluded.count),
                  confidence=MAX(relation_transitions.confidence,excluded.confidence),
                  updated_unix=MAX(relation_transitions.updated_unix,excluded.updated_unix)
                """,
                tuple(row.get(k) for k in ["key","previous_relation","next_relation","count","confidence","updated_unix","worker_id"]),
            )

    def sync(self, ram: RamSemanticMemory, role: str, pid: int, force: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        export = ram.export_records(since=self.last_export_unix)
        imported_payload = {"decisions": [], "knowledge": [], "transitions": []}
        stats = {"exported": sum(len(v) for v in export.values()), "merged": 0, "imported": 0}
        with self.lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._merge_decisions(export["decisions"])
                self._merge_knowledge(export["knowledge"])
                self._merge_transitions(export["transitions"])
                shared_counts = {
                    "decisions": self.conn.execute("SELECT COUNT(*) FROM semantic_decisions").fetchone()[0],
                    "knowledge": self.conn.execute("SELECT COUNT(*) FROM semantic_knowledge").fetchone()[0],
                    "transitions": self.conn.execute("SELECT COUNT(*) FROM relation_transitions").fetchone()[0],
                }
                imported_payload["decisions"] = [dict(r) for r in self.conn.execute("SELECT * FROM semantic_decisions WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT 5000", (self.last_import_unix,)).fetchall()]
                imported_payload["knowledge"] = [dict(r) for r in self.conn.execute("SELECT * FROM semantic_knowledge WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT 5000", (self.last_import_unix,)).fetchall()]
                imported_payload["transitions"] = [dict(r) for r in self.conn.execute("SELECT * FROM relation_transitions WHERE updated_unix > ? ORDER BY updated_unix ASC LIMIT 5000", (self.last_import_unix,)).fetchall()]
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        imp = ram.import_records(imported_payload)
        stats["imported"] = sum(imp.values())
        now_cursor = time.time()
        self.last_export_unix = max(self.last_export_unix, now_cursor)
        self.last_import_unix = max(self.last_import_unix, now_cursor)
        stats["shared_counts"] = shared_counts
        duration = time.perf_counter() - started
        self.conn.execute("INSERT INTO checkpoint_events(worker_id,event,ts,duration_s,payload_json) VALUES(?,?,?,?,?)", (self.worker_id, "checkpoint_sync", time.time(), duration, json.dumps(stats, ensure_ascii=False, sort_keys=True)))
        self.heartbeat(role, pid, last_sync=time.time())
        self.conn.commit()
        stats["duration_s"] = duration
        return stats

    def record_event(self, event: str, payload: dict[str, Any], duration_s: float = 0.0):
        with self.lock:
            self.conn.execute("INSERT INTO checkpoint_events(worker_id,event,ts,duration_s,payload_json) VALUES(?,?,?,?,?)", (self.worker_id, str(event), time.time(), float(duration_s), json.dumps(payload, ensure_ascii=False, sort_keys=True)))
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()


class SharedDistilledMemory:
    """RAM-first online semantic memory with durable graph fallback and shared checkpointing."""

    def __init__(self, graph, ram: RamSemanticMemory):
        self.graph_memory = __import__("v671_semantic_core", fromlist=["DistilledMemory"]).DistilledMemory(graph)
        self.ram = ram

    def lookup(self, decision_type, surface, context_text, candidates):
        hit = self.ram.lookup(decision_type, surface, context_text, candidates)
        if hit:
            return hit
        hit = self.graph_memory.lookup(decision_type, surface, context_text, candidates)
        if hit:
            self.ram.learn(decision_type, surface, context_text, candidates, hit["selected"], hit["confidence"], source="durable_graph_import")
        return hit

    def learn(self, decision_type, surface, context_text, candidates, selected, confidence):
        self.ram.learn(decision_type, surface, context_text, candidates, selected, confidence, source="online")
        return self.graph_memory.learn(decision_type, surface, context_text, candidates, selected, confidence)

    def goal_lookup(self, frame_key, candidates, min_confidence=0.85, min_count=2):
        hit = self.ram.goal_lookup(frame_key, candidates, min_confidence=min_confidence, min_count=min_count)
        if hit:
            return hit
        # Recover older durable goal decisions if any are present under the same
        # semantic-goal decision key. The legacy store may not have used this type.
        hit = self.graph_memory.lookup("semantic_goal", frame_key, frame_key, candidates)
        if hit and hit["count"] >= min_count and hit["confidence"] >= min_confidence:
            self.ram.learn("semantic_goal", frame_key, frame_key, candidates, hit["selected"], hit["confidence"], source="durable_graph_import")
            return {**hit, "source": "durable_graph_memory"}
        return None

    def goal_learn(self, frame_key, candidates, selected, confidence):
        self.ram.goal_learn(frame_key, candidates, selected, confidence)
        # Keep the old durable mechanism alive so prior V665/V666-style runs remain useful.
        return self.graph_memory.learn("semantic_goal", frame_key, frame_key, candidates, selected, confidence)

    def counts(self):
        c = self.ram.counts()
        legacy = self.graph_memory.counts()
        c["durable_decisions"] = int(legacy.get("decisions", 0))
        c["durable_observations"] = int(legacy.get("observations", 0))
        return c
