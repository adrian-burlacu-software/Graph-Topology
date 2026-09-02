from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


class RamSemanticMemory:
    """Per-process semantic working memory.

    V673 distinguishes local evidence from imported shared state. Imported rows
    are never exported merely because they were imported. Every locally-created
    record carries provenance and a derivation depth so the shared store can
    arbitrate evidence instead of treating every observation as graph truth.
    """

    def __init__(self, worker_id: int):
        self.worker_id = int(worker_id)
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.executescript("""
        PRAGMA journal_mode=MEMORY;
        PRAGMA synchronous=OFF;
        CREATE TABLE semantic_decisions(
            key TEXT PRIMARY KEY, decision_type TEXT NOT NULL, surface TEXT NOT NULL,
            context_key TEXT NOT NULL, candidate_set TEXT NOT NULL, selected TEXT NOT NULL,
            count INTEGER NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL,
            provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0,
            first_seen REAL NOT NULL, updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
        );
        CREATE TABLE semantic_knowledge(
            key TEXT PRIMARY KEY, kind TEXT NOT NULL, subject TEXT, relation TEXT, object TEXT,
            feature_json TEXT NOT NULL, positive INTEGER NOT NULL DEFAULT 0,
            negative INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL, provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0,
            contradiction_group TEXT, promotion_state TEXT NOT NULL DEFAULT 'candidate',
            updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
        );
        CREATE TABLE relation_transitions(
            key TEXT PRIMARY KEY, previous_relation TEXT NOT NULL, next_relation TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0.0,
            provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0,
            updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL
        );
        CREATE TABLE worker_local_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.conn.commit()

    def counts(self):
        with self.lock:
            return {k: self.conn.execute(f"SELECT COUNT(*) FROM {k}").fetchone()[0]
                    for k in ("semantic_decisions", "semantic_knowledge", "relation_transitions")}

    def provenance_counts(self):
        with self.lock:
            out = {}
            for table in ("semantic_decisions", "semantic_knowledge", "relation_transitions"):
                try:
                    rows = self.conn.execute(f"SELECT provenance,COUNT(*) n FROM {table} GROUP BY provenance").fetchall()
                    out[table] = {str(r[0]): int(r[1]) for r in rows}
                except sqlite3.OperationalError:
                    out[table] = {}
            return out

    @staticmethod
    def _candidate_json(candidates):
        return json.dumps(sorted({str(x) for x in candidates}), ensure_ascii=False, sort_keys=True)

    def lookup(self, decision_type, surface, context_text, candidates):
        if not candidates:
            return None
        context_key = _stable_key(decision_type, context_text)
        candidate_json = self._candidate_json(candidates)
        with self.lock:
            row = self.conn.execute("""SELECT selected,count,confidence,source,provenance,candidate_set
                FROM semantic_decisions WHERE decision_type=? AND context_key=? AND candidate_set=?
                ORDER BY count DESC, confidence DESC, updated_unix DESC LIMIT 1""",
                (str(decision_type), context_key, candidate_json)).fetchone()
        if not row or str(row["selected"]) not in {str(x) for x in candidates}:
            return None
        return {"selected": str(row["selected"]), "count": int(row["count"]),
                "confidence": float(row["confidence"]), "source": str(row["source"]),
                "provenance": str(row["provenance"])}

    def learn(self, decision_type, surface, context_text, candidates, selected, confidence,
              source="online", provenance="observed", derivation_depth=0):
        if not candidates or selected not in candidates:
            return False
        now = time.time(); context_key = _stable_key(decision_type, context_text)
        candidate_json = self._candidate_json(candidates)
        key = _stable_key("decision", decision_type, surface, context_key, candidate_json, selected)
        confidence = max(0.0, min(1.0, float(confidence)))
        with self.lock:
            row = self.conn.execute("SELECT count,confidence FROM semantic_decisions WHERE key=?", (key,)).fetchone()
            if row:
                count = int(row["count"]) + 1
                avg = ((float(row["confidence"]) * (count - 1)) + confidence) / count
                self.conn.execute("UPDATE semantic_decisions SET count=?,confidence=?,updated_unix=?,source=?,provenance=?,derivation_depth=?,worker_id=? WHERE key=?",
                                  (count, avg, now, source, provenance, int(derivation_depth), self.worker_id, key))
            else:
                self.conn.execute("""INSERT INTO semantic_decisions
                    (key,decision_type,surface,context_key,candidate_set,selected,count,confidence,source,provenance,derivation_depth,first_seen,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (key,str(decision_type),str(surface),context_key,candidate_json,str(selected),1,confidence,source,provenance,int(derivation_depth),now,now,self.worker_id))
            self.conn.commit()
        return True

    def goal_lookup(self, frame_key, candidates, min_confidence=0.85, min_count=2):
        result = self.lookup("semantic_goal", frame_key, frame_key, candidates)
        if not result or result["count"] < int(min_count) or result["confidence"] < float(min_confidence):
            return None
        result["source"] = "ram_semantic_memory"
        return result

    def goal_learn(self, frame_key, candidates, selected, confidence):
        return self.learn("semantic_goal", frame_key, frame_key, candidates, selected, confidence,
                          source="online_goal", provenance="observed", derivation_depth=0)

    def upsert_knowledge(self, kind, subject, relation, obj, feature, positive=0, negative=0,
                         confidence=0.0, source="offline", provenance="observed", derivation_depth=0,
                         contradiction_group=None, promotion_state="candidate"):
        now=time.time(); feature_json=json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(",",":"))
        key=_stable_key("knowledge",kind,subject,relation,obj,feature_json)
        with self.lock:
            row=self.conn.execute("SELECT positive,negative,confidence FROM semantic_knowledge WHERE key=?",(key,)).fetchone()
            if row:
                p=int(row["positive"])+int(positive); n=int(row["negative"])+int(negative)
                total=p+n; c0=float(row["confidence"])
                c=((c0*max(total-int(positive)-int(negative),0))+float(confidence)*max(int(positive)+int(negative),1))/max(total,1)
                self.conn.execute("""UPDATE semantic_knowledge SET positive=?,negative=?,confidence=?,source=?,provenance=?,derivation_depth=?,contradiction_group=?,promotion_state=?,updated_unix=?,worker_id=? WHERE key=?""",
                                  (p,n,max(0,min(1,c)),source,provenance,int(derivation_depth),contradiction_group,promotion_state,now,self.worker_id,key))
            else:
                self.conn.execute("""INSERT INTO semantic_knowledge
                    (key,kind,subject,relation,object,feature_json,positive,negative,confidence,source,provenance,derivation_depth,contradiction_group,promotion_state,updated_unix,worker_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (key,kind,subject,relation,obj,feature_json,int(positive),int(negative),max(0,min(1,float(confidence))),source,provenance,int(derivation_depth),contradiction_group,promotion_state,now,self.worker_id))
            self.conn.commit()
        return key

    def upsert_transition(self, previous_relation, next_relation, confidence=0.0, count=1,
                          provenance="observed", derivation_depth=0):
        key=_stable_key("transition",previous_relation,next_relation); now=time.time()
        with self.lock:
            row=self.conn.execute("SELECT count,confidence FROM relation_transitions WHERE key=?",(key,)).fetchone()
            if row:
                old=int(row["count"]); n=old+int(count)
                avg=((float(row["confidence"])*old)+(float(confidence)*int(count)))/max(n,1)
                self.conn.execute("UPDATE relation_transitions SET count=?,confidence=?,provenance=?,derivation_depth=?,updated_unix=?,worker_id=? WHERE key=?",
                                  (n,avg,provenance,int(derivation_depth),now,self.worker_id,key))
            else:
                self.conn.execute("INSERT INTO relation_transitions VALUES(?,?,?,?,?,?,?,?,?)",
                                  (key,str(previous_relation),str(next_relation),int(count),float(confidence),provenance,int(derivation_depth),now,self.worker_id))
            self.conn.commit()
        return key

    def export_records(self, limit=5000, since=0.0):
        """Export only records whose current owner is this worker.
        This prevents imported aggregate state from being re-exported as fresh evidence.
        """
        with self.lock:
            args=(self.worker_id,float(since),int(limit))
            decisions=[dict(r) for r in self.conn.execute("SELECT * FROM semantic_decisions WHERE worker_id=? AND updated_unix>? ORDER BY updated_unix ASC LIMIT ?",args).fetchall()]
            knowledge=[dict(r) for r in self.conn.execute("SELECT * FROM semantic_knowledge WHERE worker_id=? AND updated_unix>? ORDER BY updated_unix ASC LIMIT ?",args).fetchall()]
            transitions=[dict(r) for r in self.conn.execute("SELECT * FROM relation_transitions WHERE worker_id=? AND updated_unix>? ORDER BY updated_unix ASC LIMIT ?",args).fetchall()]
        return {"decisions":decisions,"knowledge":knowledge,"transitions":transitions}

    def import_records(self, payload):
        imported={"decisions":0,"knowledge":0,"transitions":0}
        with self.lock:
            for row in payload.get("decisions",[]):
                self.conn.execute("""INSERT INTO semantic_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET count=MAX(semantic_decisions.count,excluded.count),confidence=MAX(semantic_decisions.confidence,excluded.confidence),updated_unix=MAX(semantic_decisions.updated_unix,excluded.updated_unix),source=excluded.source,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth""",
                    tuple(row.get(k) for k in ["key","decision_type","surface","context_key","candidate_set","selected","count","confidence","source","provenance","derivation_depth","first_seen","updated_unix","worker_id"]))
                imported["decisions"]+=1
            for row in payload.get("knowledge",[]):
                self.conn.execute("""INSERT INTO semantic_knowledge VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET positive=MAX(semantic_knowledge.positive,excluded.positive),negative=MAX(semantic_knowledge.negative,excluded.negative),confidence=MAX(semantic_knowledge.confidence,excluded.confidence),updated_unix=MAX(semantic_knowledge.updated_unix,excluded.updated_unix),source=excluded.source,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth,contradiction_group=excluded.contradiction_group,promotion_state=excluded.promotion_state,worker_id=excluded.worker_id""",
                    tuple(row.get(k) for k in ["key","kind","subject","relation","object","feature_json","positive","negative","confidence","source","provenance","derivation_depth","contradiction_group","promotion_state","updated_unix","worker_id"]))
                imported["knowledge"]+=1
            for row in payload.get("transitions",[]):
                self.conn.execute("""INSERT INTO relation_transitions VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET count=MAX(relation_transitions.count,excluded.count),confidence=MAX(relation_transitions.confidence,excluded.confidence),updated_unix=MAX(relation_transitions.updated_unix,excluded.updated_unix),provenance=excluded.provenance,derivation_depth=excluded.derivation_depth,worker_id=excluded.worker_id""",
                    tuple(row.get(k) for k in ["key","previous_relation","next_relation","count","confidence","provenance","derivation_depth","updated_unix","worker_id"]))
                imported["transitions"]+=1
            self.conn.commit()
        return imported

    def relation_prior(self, previous_relations, candidate_relation):
        if not previous_relations:return 0.0
        with self.lock:
            vals=[]
            for prev in previous_relations[-3:]:
                row=self.conn.execute("SELECT count,confidence FROM relation_transitions WHERE previous_relation=? AND next_relation=?",(str(prev),str(candidate_relation))).fetchone()
                if row: vals.append(min(1,float(row["confidence"])) * min(1,float(row["count"])/10))
        return max(vals,default=0.0)


class SharedCheckpoint:
    """Shared WAL store with per-worker evidence contributions and arbitration."""
    def __init__(self,path,worker_id,total_workers,interval_seconds=300):
        self.path=Path(path); self.worker_id=int(worker_id); self.total_workers=int(total_workers); self.interval_seconds=int(interval_seconds)
        if self.interval_seconds < self.total_workers: raise ValueError("checkpoint interval must be >= total worker count")
        self.lock=threading.RLock(); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(str(self.path),timeout=10,check_same_thread=False); self.conn.row_factory=sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=10000"); self.conn.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS semantic_decisions(
          key TEXT PRIMARY KEY, decision_type TEXT NOT NULL, surface TEXT NOT NULL, context_key TEXT NOT NULL,
          candidate_set TEXT NOT NULL, selected TEXT NOT NULL, count INTEGER NOT NULL, confidence REAL NOT NULL,
          source TEXT NOT NULL, provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0,
          first_seen REAL NOT NULL, updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS semantic_knowledge(
          key TEXT PRIMARY KEY, kind TEXT NOT NULL, subject TEXT, relation TEXT, object TEXT, feature_json TEXT NOT NULL,
          positive INTEGER NOT NULL, negative INTEGER NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL,
          provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0, contradiction_group TEXT,
          promotion_state TEXT NOT NULL DEFAULT 'candidate', updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS relation_transitions(
          key TEXT PRIMARY KEY, previous_relation TEXT NOT NULL, next_relation TEXT NOT NULL, count INTEGER NOT NULL,
          confidence REAL NOT NULL, provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL DEFAULT 0,
          updated_unix REAL NOT NULL, worker_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS decision_evidence(
          key TEXT NOT NULL, worker_id INTEGER NOT NULL, count INTEGER NOT NULL, confidence REAL NOT NULL,
          selected TEXT NOT NULL, updated_unix REAL NOT NULL, source TEXT NOT NULL, provenance TEXT NOT NULL,
          derivation_depth INTEGER NOT NULL, PRIMARY KEY(key,worker_id));
        CREATE TABLE IF NOT EXISTS knowledge_evidence(
          key TEXT NOT NULL, worker_id INTEGER NOT NULL, positive INTEGER NOT NULL, negative INTEGER NOT NULL,
          confidence REAL NOT NULL, source TEXT NOT NULL, provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL,
          contradiction_group TEXT, updated_unix REAL NOT NULL, PRIMARY KEY(key,worker_id));
        CREATE TABLE IF NOT EXISTS transition_evidence(
          key TEXT NOT NULL, worker_id INTEGER NOT NULL, count INTEGER NOT NULL, confidence REAL NOT NULL,
          provenance TEXT NOT NULL, derivation_depth INTEGER NOT NULL, updated_unix REAL NOT NULL, PRIMARY KEY(key,worker_id));
        CREATE TABLE IF NOT EXISTS decision_resolution(
          group_key TEXT PRIMARY KEY, decision_type TEXT NOT NULL, context_key TEXT NOT NULL, candidate_set TEXT NOT NULL,
          selected TEXT NOT NULL, support_count INTEGER NOT NULL, confidence REAL NOT NULL, contested INTEGER NOT NULL,
          updated_unix REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS worker_status(
          worker_id INTEGER PRIMARY KEY, role TEXT NOT NULL, pid INTEGER, last_seen REAL NOT NULL, last_sync REAL,
          batches INTEGER NOT NULL DEFAULT 0, items INTEGER NOT NULL DEFAULT 0, learned INTEGER NOT NULL DEFAULT 0,
          imported INTEGER NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0, last_error TEXT,
          last_batch_s REAL DEFAULT 0, learned_per_s REAL DEFAULT 0, last_sync_s REAL DEFAULT 0, sync_count INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS checkpoint_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id INTEGER NOT NULL, event TEXT NOT NULL, ts REAL NOT NULL,
          duration_s REAL NOT NULL, payload_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_checkpoint_events_ts ON checkpoint_events(ts);
        CREATE TABLE IF NOT EXISTS merge_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id INTEGER NOT NULL, table_name TEXT NOT NULL, key TEXT NOT NULL,
          action TEXT NOT NULL, conflict INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL, payload_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_merge_events_ts ON merge_events(ts);
        """); self.conn.commit()
        self.last_bucket=None; self.last_export_unix=0.0; self.last_import_unix=0.0

    def target_slot(self):
        # Evenly distribute workers over the ENTIRE interval: 5 min => ~15s apart;
        # 1 min => ~3s apart. The old V671 mapping packed all 20 into the first 20s.
        return int((self.worker_id * self.interval_seconds) / self.total_workers)

    def should_sync(self, now=None):
        now=time.time() if now is None else float(now); bucket=int(now)//self.interval_seconds
        if bucket==self.last_bucket:return False
        elapsed=now-(bucket*self.interval_seconds); target=self.target_slot()
        if abs(elapsed-target) <= 0.75:
            self.last_bucket=bucket; return True
        return False

    def heartbeat(self,role,pid,**updates):
        with self.lock:
            prev=self.conn.execute("SELECT * FROM worker_status WHERE worker_id=?",(self.worker_id,)).fetchone()
            values={"role":role,"pid":int(pid),"last_seen":time.time(),"last_sync":updates.get("last_sync",prev["last_sync"] if prev else None),
                    "batches":int(updates.get("batches",prev["batches"] if prev else 0)),"items":int(updates.get("items",prev["items"] if prev else 0)),
                    "learned":int(updates.get("learned",prev["learned"] if prev else 0)),"imported":int(updates.get("imported",prev["imported"] if prev else 0)),
                    "errors":int(updates.get("errors",prev["errors"] if prev else 0)),"last_error":updates.get("last_error",prev["last_error"] if prev else None),
                    "last_batch_s":float(updates.get("last_batch_s",prev["last_batch_s"] if prev else 0)),"learned_per_s":float(updates.get("learned_per_s",prev["learned_per_s"] if prev else 0)),
                    "last_sync_s":float(updates.get("last_sync_s",prev["last_sync_s"] if prev else 0)),"sync_count":int(updates.get("sync_count",prev["sync_count"] if prev else 0))}
            if prev:
                self.conn.execute("""UPDATE worker_status SET role=?,pid=?,last_seen=?,last_sync=?,batches=?,items=?,learned=?,imported=?,errors=?,last_error=?,last_batch_s=?,learned_per_s=?,last_sync_s=?,sync_count=? WHERE worker_id=?""",(*values.values(),self.worker_id))
            else:
                self.conn.execute("""INSERT INTO worker_status(worker_id,role,pid,last_seen,last_sync,batches,items,learned,imported,errors,last_error,last_batch_s,learned_per_s,last_sync_s,sync_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(self.worker_id,*values.values()))
            self.conn.commit()

    def _merge(self, ram, export):
        conflicts=0; merged=0
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for row in export["decisions"]:
                key=row["key"]; wid=int(row["worker_id"])
                group_key=_stable_key("decision_group",row["decision_type"],row["context_key"],row["candidate_set"])
                self.conn.execute("""INSERT INTO decision_evidence VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key,worker_id) DO UPDATE SET count=excluded.count,confidence=excluded.confidence,selected=excluded.selected,updated_unix=excluded.updated_unix,source=excluded.source,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth""",
                    (group_key,wid,int(row["count"]),float(row["confidence"]),row["selected"],float(row["updated_unix"]),row["source"],row["provenance"],int(row["derivation_depth"])))
                vals=self.conn.execute("SELECT * FROM decision_evidence WHERE key=?",(group_key,)).fetchall()
                selecteds={str(v["selected"]) for v in vals}; conflicts += int(len(selecteds)>1); merged += int(len(vals)>1)
                winner=max(vals,key=lambda v:(int(v["count"]),float(v["confidence"]),float(v["updated_unix"])))
                self.conn.execute("""INSERT INTO semantic_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET count=excluded.count,confidence=excluded.confidence,selected=excluded.selected,source=excluded.source,provenance='arbitrated',derivation_depth=excluded.derivation_depth,updated_unix=excluded.updated_unix,worker_id=excluded.worker_id""",
                    (key,row["decision_type"],row["surface"],row["context_key"],row["candidate_set"],winner["selected"],sum(int(v["count"]) for v in vals),sum(float(v["confidence"])*int(v["count"]) for v in vals)/max(sum(int(v["count"]) for v in vals),1),winner["source"],"arbitrated",max(int(v["derivation_depth"]) for v in vals),float(row["first_seen"]),max(float(v["updated_unix"]) for v in vals),int(winner["worker_id"])))
                total_support=sum(int(v["count"]) for v in vals)
                win_conf=sum(float(v["confidence"])*int(v["count"]) for v in vals)/max(total_support,1)
                contested=int(len(selecteds)>1)
                self.conn.execute("""INSERT INTO decision_resolution VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(group_key) DO UPDATE SET selected=excluded.selected,support_count=excluded.support_count,confidence=excluded.confidence,contested=excluded.contested,updated_unix=excluded.updated_unix""",
                    (group_key,row["decision_type"],row["context_key"],row["candidate_set"],winner["selected"],total_support,win_conf,contested,max(float(v["updated_unix"]) for v in vals)))
            for row in export["knowledge"]:
                key=row["key"]; wid=int(row["worker_id"])
                self.conn.execute("""INSERT INTO knowledge_evidence VALUES(?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(key,worker_id) DO UPDATE SET positive=excluded.positive,negative=excluded.negative,confidence=excluded.confidence,source=excluded.source,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth,contradiction_group=excluded.contradiction_group,updated_unix=excluded.updated_unix""",
                  (key,wid,int(row["positive"]),int(row["negative"]),float(row["confidence"]),row["source"],row["provenance"],int(row["derivation_depth"]),row["contradiction_group"],float(row["updated_unix"])))
                vals=self.conn.execute("SELECT * FROM knowledge_evidence WHERE key=?",(key,)).fetchall(); merged += int(len(vals)>1)
                p=sum(int(v["positive"]) for v in vals); n=sum(int(v["negative"]) for v in vals); total=p+n
                weighted=sum(float(v["confidence"])*(int(v["positive"])+int(v["negative"])) for v in vals)/max(total,1)
                conflict=int(p>0 and n>0); conflicts += conflict
                best=max(vals,key=lambda v:(float(v["confidence"]),int(v["positive"])-int(v["negative"])))
                promotion="candidate"
                if conflict: promotion="contested"
                elif total>=3 and weighted>=0.85 and int(best["derivation_depth"])==0: promotion="eligible"
                self.conn.execute("""INSERT INTO semantic_knowledge VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET positive=excluded.positive,negative=excluded.negative,confidence=excluded.confidence,source=excluded.source,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth,contradiction_group=excluded.contradiction_group,promotion_state=excluded.promotion_state,updated_unix=excluded.updated_unix,worker_id=excluded.worker_id""",
                  (key,row["kind"],row["subject"],row["relation"],row["object"],row["feature_json"],p,n,max(0,min(1,weighted)),best["source"],"arbitrated",max(int(v["derivation_depth"]) for v in vals),best["contradiction_group"],promotion,max(float(v["updated_unix"]) for v in vals),int(best["worker_id"])))
            for row in export["transitions"]:
                key=row["key"]; wid=int(row["worker_id"])
                self.conn.execute("""INSERT INTO transition_evidence VALUES(?,?,?,?,?,?,?) ON CONFLICT(key,worker_id) DO UPDATE SET count=excluded.count,confidence=excluded.confidence,provenance=excluded.provenance,derivation_depth=excluded.derivation_depth,updated_unix=excluded.updated_unix""",
                  (key,wid,int(row["count"]),float(row["confidence"]),row["provenance"],int(row["derivation_depth"]),float(row["updated_unix"])))
                vals=self.conn.execute("SELECT * FROM transition_evidence WHERE key=?",(key,)).fetchall(); merged += int(len(vals)>1)
                total=sum(int(v["count"]) for v in vals); conf=sum(float(v["confidence"])*int(v["count"]) for v in vals)/max(total,1); best=max(vals,key=lambda v:float(v["updated_unix"]))
                self.conn.execute("""INSERT INTO relation_transitions VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET count=excluded.count,confidence=excluded.confidence,provenance='arbitrated',derivation_depth=excluded.derivation_depth,updated_unix=excluded.updated_unix,worker_id=?""",
                  (key,row["previous_relation"],row["next_relation"],total,conf,"arbitrated",max(int(v["derivation_depth"]) for v in vals),max(float(v["updated_unix"]) for v in vals),int(best["worker_id"]),int(best["worker_id"])))
            self.conn.commit()
        except Exception:
            self.conn.rollback(); raise
        return merged,conflicts

    def sync(self,ram,role,pid,force=False):
        started=time.perf_counter(); export=ram.export_records(since=self.last_export_unix)
        stats={"exported":sum(len(v) for v in export.values()),"merged":0,"conflicts":0,"imported":0}
        with self.lock:
            merged,conflicts=self._merge(ram,export); stats["merged"]=merged; stats["conflicts"]=conflicts
            now=time.time()
            # Import only aggregate rows changed since last import. The import is
            # explicitly marked shared/arbitrated, and does not become worker evidence.
            payload={k:[dict(r) for r in self.conn.execute(f"SELECT * FROM {t} WHERE updated_unix>? ORDER BY updated_unix ASC LIMIT 5000",(self.last_import_unix,)).fetchall()]
                     for k,t in [("decisions","semantic_decisions"),("knowledge","semantic_knowledge"),("transitions","relation_transitions")]}
            shared_counts={t:self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("semantic_decisions","semantic_knowledge","relation_transitions")}
            self.last_export_unix=max(self.last_export_unix,now); self.last_import_unix=max(self.last_import_unix,now)
            duration=time.perf_counter()-started
            stats["imported"]=sum(ram.import_records(payload).values())
            stats["shared_counts"]={"decisions":shared_counts["semantic_decisions"],"knowledge":shared_counts["semantic_knowledge"],"transitions":shared_counts["relation_transitions"]}
            stats["duration_s"]=duration
            stats["target_slot_s"]=self.target_slot(); stats["force"]=bool(force)
            self.conn.execute("INSERT INTO checkpoint_events(worker_id,event,ts,duration_s,payload_json) VALUES(?,?,?,?,?)",(self.worker_id,"checkpoint_sync",now,duration,json.dumps(stats,ensure_ascii=False,sort_keys=True)))
            self.conn.commit()
        self.heartbeat(role,pid,last_sync=time.time(),last_sync_s=duration,sync_count=self._sync_count()+1)
        return stats

    def goal_resolution(self,frame_key,candidates,min_confidence=0.85,min_count=2):
        candidate_json=json.dumps(sorted({str(x) for x in candidates}),ensure_ascii=False,sort_keys=True)
        context_key=_stable_key("semantic_goal",frame_key)
        row=self.conn.execute("SELECT selected,support_count,confidence,contested FROM decision_resolution WHERE decision_type=? AND context_key=? AND candidate_set=? LIMIT 1",("semantic_goal",context_key,candidate_json)).fetchone()
        if not row or int(row["contested"]) or int(row["support_count"])<int(min_count) or float(row["confidence"])<float(min_confidence):
            return None
        if str(row["selected"]) not in {str(x) for x in candidates}: return None
        return {"selected":str(row["selected"]),"count":int(row["support_count"]),"confidence":float(row["confidence"]),"source":"shared_arbitration","provenance":"arbitrated"}

    def _sync_count(self):
        row=self.conn.execute("SELECT sync_count FROM worker_status WHERE worker_id=?",(self.worker_id,)).fetchone(); return int(row[0]) if row else 0

    def record_event(self,event,payload,duration_s=0.0):
        with self.lock:
            self.conn.execute("INSERT INTO checkpoint_events(worker_id,event,ts,duration_s,payload_json) VALUES(?,?,?,?,?)",(self.worker_id,str(event),time.time(),float(duration_s),json.dumps(payload,ensure_ascii=False,sort_keys=True))); self.conn.commit()

    def close(self):
        with self.lock:self.conn.close()


class SharedDistilledMemory:
    def __init__(self,graph,ram,checkpoint=None):
        self.graph_memory=__import__("v673_semantic_core",fromlist=["DistilledMemory"]).DistilledMemory(graph); self.ram=ram; self.checkpoint=checkpoint
    def lookup(self,decision_type,surface,context_text,candidates):
        hit=self.ram.lookup(decision_type,surface,context_text,candidates)
        if hit:return hit
        hit=self.graph_memory.lookup(decision_type,surface,context_text,candidates)
        if hit:self.ram.learn(decision_type,surface,context_text,candidates,hit["selected"],hit["confidence"],source="durable_graph_import",provenance="observed")
        return hit
    def learn(self,decision_type,surface,context_text,candidates,selected,confidence):
        self.ram.learn(decision_type,surface,context_text,candidates,selected,confidence,source="online",provenance="observed")
        return self.graph_memory.learn(decision_type,surface,context_text,candidates,selected,confidence)
    def goal_lookup(self,frame_key,candidates,min_confidence=0.85,min_count=2):
        # Prefer the shared arbitration result when available. This prevents a
        # tied local candidate from being selected merely because it arrived last.
        if self.checkpoint is not None:
            try:
                hit=self.checkpoint.goal_resolution(frame_key,candidates,min_confidence,min_count)
                if hit:return hit
            except Exception:
                pass
        hit=self.ram.goal_lookup(frame_key,candidates,min_confidence,min_count)
        if hit:return hit
        hit=self.graph_memory.lookup("semantic_goal",frame_key,frame_key,candidates)
        if hit and hit["count"]>=min_count and hit["confidence"]>=min_confidence:
            self.ram.learn("semantic_goal",frame_key,frame_key,candidates,hit["selected"],hit["confidence"],source="durable_graph_import",provenance="observed")
            return {**hit,"source":"durable_graph_memory"}
        return None
    def goal_learn(self,frame_key,candidates,selected,confidence):
        self.ram.goal_learn(frame_key,candidates,selected,confidence)
        return self.graph_memory.learn("semantic_goal",frame_key,frame_key,candidates,selected,confidence)
    def counts(self):
        c=self.ram.counts(); legacy=self.graph_memory.counts(); c["durable_decisions"]=int(legacy.get("decisions",0)); c["durable_observations"]=int(legacy.get("observations",0)); return c
