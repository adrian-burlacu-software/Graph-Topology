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
from collections import Counter, defaultdict
from dataclasses import dataclass

DEFAULT_WORKERS = 20

POSITIVE_FAMILIES = {
    "is_a": "classification", "instance_of": "classification", "defined_as": "definition",
    "has_property": "property", "has": "possession", "has_part": "composition",
    "part_of": "composition", "capable_of": "capability", "causes": "causality",
    "located_in": "location", "used_for": "purpose", "made_of": "composition",
}

# Actual negative predicates observed in this graph family, plus common aliases.
NEGATIVE_FAMILIES = {
    "notcapableof": ("capability", "cannot"),
    "not_capable_of": ("capability", "cannot"),
    "nothasproperty": ("property", "not_property"),
    "not_has_property": ("property", "not_property"),
    "nothas": ("possession", "not_possession"),
    "not_has": ("possession", "not_possession"),
    "notdesires": ("desire", "not_desire"),
    "not_desires": ("desire", "not_desire"),
    "have_not": ("possession", "not_possession"),
    "has_not": ("possession", "not_possession"),
    "not_has_part": ("composition", "not_composition"),
    "notpartof": ("composition", "not_composition"),
}

NEGATION_TO_POSITIVE = {
    "notcapableof": "capable_of", "not_capable_of": "capable_of",
    "nothasproperty": "has_property", "not_has_property": "has_property",
    "nothas": "has", "not_has": "has", "have_not": "has", "has_not": "has",
    "not_has_part": "has_part", "notpartof": "part_of",
    "notdesires": "desires", "not_desires": "desires",
}

BLOCKED_PREDICATES = {
    "in_domain", "domain", "source", "provenance", "dataset", "node_type", "type",
    "label", "nsubj", "nsubjpass", "obj", "dobj", "iobj", "ccomp", "xcomp", "amod",
    "advmod", "nmod", "obl", "oblique", "root", "dep", "aux", "auxpass", "cop", "det",
    "case", "mark", "punct", "conj", "cc", "compound", "appos", "acl", "advcl",
}
EXCLUDED_TABLE_WORDS = {"live", "conversation", "session", "turn", "history", "message", "utterance"}
GENERIC_WORDS = {"entity", "thing", "object", "concept", "something", "someone", "word", "term"}

@dataclass(frozen=True)
class KnowledgeAdapter:
    mode: str
    table: str
    concepts: str | None = None

    def base_from(self) -> str:
        if self.mode == "canonical":
            return f'''FROM "{self.table}" f
LEFT JOIN "{self.concepts}" cs ON cs.concept_id=f.subject_id
LEFT JOIN "{self.concepts}" co ON co.concept_id=f.object_id'''
        return f'FROM "{self.table}" f'

    def subject_expr(self) -> str:
        return "cs.canonical" if self.mode == "canonical" else "f.subject"

    def object_expr(self) -> str:
        return "COALESCE(co.canonical, f.object_text)" if self.mode == "canonical" else "f.object_text"

CURRENT_SCHEMA: dict[str, list[str]] = {}

def connect(path: str) -> sqlite3.Connection:
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA mmap_size=1073741824")
    con.execute("PRAGMA cache_size=-131072")
    return con

def inspect_schema(con):
    global CURRENT_SCHEMA
    info = {}
    for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        table = r[0]
        info[table] = [x[1] for x in con.execute(f'PRAGMA table_info("{table}")')]
    CURRENT_SCHEMA = info
    return info

def table_rows(con, table):
    try:
        return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except sqlite3.Error:
        return 0

def pick_adapter(con, info, forced):
    candidates = []
    for table, cols_list in info.items():
        cols = set(cols_list)
        low = table.lower()
        excluded = any(w in low for w in EXCLUDED_TABLE_WORDS)
        rows = table_rows(con, table)
        if {"subject_id", "predicate", "object_id", "object_text"}.issubset(cols) and "facts" in low and "concepts" in info and {"concept_id", "canonical"}.issubset(set(info["concepts"])):
            candidates.append({"table": table, "mode": "canonical", "rows": rows, "excluded": excluded, "score": rows + 10_000_000})
            continue
        if {"subject", "predicate", "object_text"}.issubset(cols):
            score = rows + (1_000_000 if any(x in low for x in ("fact", "edge", "knowledge", "semantic")) else 0)
            if excluded:
                score -= 5_000_000
            candidates.append({"table": table, "mode": "flat", "rows": rows, "excluded": excluded, "score": score})
    if forced:
        if forced not in info:
            raise RuntimeError(f"Requested --table {forced!r} does not exist")
        cols = set(info[forced])
        if {"subject_id", "predicate", "object_id", "object_text"}.issubset(cols) and "concepts" in info:
            return KnowledgeAdapter("canonical", forced, "concepts"), candidates
        if {"subject", "predicate", "object_text"}.issubset(cols):
            return KnowledgeAdapter("flat", forced, None), candidates
        raise RuntimeError(f"Table {forced!r} is not supported")
    viable = [c for c in candidates if not c["excluded"] and c["rows"] > 0]
    if not viable:
        raise RuntimeError("No non-live knowledge graph table found: " + ", ".join(sorted(info)))
    chosen = max(viable, key=lambda x: x["score"])
    return KnowledgeAdapter(chosen["mode"], chosen["table"], "concepts" if chosen["mode"] == "canonical" else None), candidates

def relation_inventory(con, adapter):
    sql = f'SELECT predicate, COUNT(*) AS n FROM "{adapter.table}" WHERE predicate IS NOT NULL GROUP BY predicate ORDER BY n DESC'
    out = []
    for r in con.execute(sql):
        p = str(r["predicate"])
        family = POSITIVE_FAMILIES.get(p, NEGATIVE_FAMILIES.get(p, ("other", ""))[0])
        out.append({"predicate": p, "count": int(r["n"]), "family": family})
    return out

def natural_question(predicate, subject, obj):
    s, o = str(subject).strip(), str(obj).strip()
    if predicate in {"is_a", "instance_of"}: return f"Is {s} a {o}?"
    if predicate == "defined_as": return f"Is {s} defined as {o}?"
    if predicate in {"has", "has_part"}: return f"Does {s} have {o}?"
    if predicate == "part_of": return f"Is {s} part of {o}?"
    if predicate == "has_property": return f"Is {s} {o}?"
    if predicate == "capable_of": return f"Can {s} {o}?"
    if predicate == "causes": return f"Does {s} cause {o}?"
    if predicate == "located_in": return f"Is {s} located in {o}?"
    if predicate == "used_for": return f"Is {s} used for {o}?"
    if predicate == "made_of": return f"Is {s} made of {o}?"
    if predicate in NEGATION_TO_POSITIVE:
        pos = NEGATION_TO_POSITIVE[predicate]
        return natural_question(pos, s, o).replace("?", "?").replace("Does ", "Does ", 1)
    return f"Does {s} {predicate.replace('_',' ')} {o}?"

def negative_question(predicate, subject, obj):
    s, o = str(subject).strip(), str(obj).strip()
    if predicate in {"notcapableof", "not_capable_of"}: return f"Can {s} {o}?"
    if predicate in {"nothasproperty", "not_has_property"}: return f"Is {s} {o}?"
    if predicate in {"nothas", "not_has", "have_not", "has_not"}: return f"Does {s} have {o}?"
    if predicate in {"not_has_part"}: return f"Does {s} have {o}?"
    if predicate in {"notpartof"}: return f"Is {s} part of {o}?"
    if predicate in {"notdesires", "not_desires"}: return f"Does {s} desire {o}?"
    return f"Does {s} {predicate.replace('_',' ')} {o}?"

def fetch_rows(con, adapter, predicates=None, limit=1000, order=None):
    params = []
    where = [f"{adapter.subject_expr()} IS NOT NULL", f"{adapter.object_expr()} IS NOT NULL"]
    if predicates:
        where.append("f.predicate IN (" + ",".join("?" for _ in predicates) + ")")
        params.extend(predicates)
    sql = f'''SELECT {adapter.subject_expr()} AS subject, f.predicate, {adapter.object_expr()} AS object_text
              {adapter.base_from()} WHERE {' AND '.join(where)}'''
    if order:
        sql += " ORDER BY " + order
    sql += " LIMIT ?"
    params.append(int(limit))
    return [dict(r) for r in con.execute(sql, params)]

def sample_direct(con, adapter, n, seed):
    allowed = [r["predicate"] for r in relation_inventory(con, adapter) if r["predicate"] in POSITIVE_FAMILIES]
    if not allowed: return []
    rng = random.Random(seed); rng.shuffle(allowed)
    per = max(1, math.ceil(n / len(allowed)))
    out, seen = [], set()
    for p in allowed:
        for row in fetch_rows(con, adapter, [p], per):
            k = (str(row["subject"]).lower(), p, str(row["object_text"]).lower())
            if k in seen: continue
            seen.add(k)
            out.append({"question": natural_question(p,row["subject"],row["object_text"]),"status":"SUPPORTED","kind":"direct","subject":row["subject"],"predicate":p,"object":row["object_text"],"path":[[row["subject"],p,row["object_text"]]]})
            if len(out)>=n: return out
    return out

def sample_negative(con, adapter, n, seed):
    actual = [r["predicate"] for r in relation_inventory(con, adapter) if r["predicate"] in NEGATIVE_FAMILIES]
    if not actual: return []
    rows = fetch_rows(con, adapter, actual, max(n*3,n))
    rng = random.Random(seed); rng.shuffle(rows)
    out, seen = [], set()
    for row in rows:
        p = str(row["predicate"]); s=str(row["subject"]); o=str(row["object_text"])
        k=(s.lower(),p,o.lower())
        if k in seen: continue
        seen.add(k)
        fam, kind = NEGATIVE_FAMILIES[p]
        out.append({"question":negative_question(p,s,o),"status":"REFUTED","kind":"explicit_negative","subject":s,"predicate":p,"object":o,"negative_family":kind,"path":[[s,p,o]]})
        if len(out)>=n: break
    return out

def load_edges(con, adapter, predicates):
    rows = fetch_rows(con, adapter, predicates, 2_000_000)
    by_subject = defaultdict(list)
    for r in rows:
        s,o,p = str(r["subject"]),str(r["object_text"]),str(r["predicate"])
        if not s or not o or len(o)>120: continue
        if p in BLOCKED_PREDICATES: continue
        by_subject[s.lower()].append((s,p,o))
    return by_subject

def mine_indirect(con, adapter, n, max_hops, seed):
    # Read a bounded but broad semantic edge set once; no per-node SQL.
    predicates = tuple(POSITIVE_FAMILIES.keys())
    by_subject = load_edges(con, adapter, predicates)
    starts = list(by_subject.keys())
    rng = random.Random(seed); rng.shuffle(starts)
    safe = {("is_a","has_part"):"has_part", ("instance_of","has_part"):"has_part", ("is_a","has"):"has", ("instance_of","has"):"has", ("is_a","has_property"):"has_property", ("instance_of","has_property"):"has_property", ("has_part","part_of"):"has_part", ("has","part_of"):"has"}
    out=[]; seen=set()
    for start_key in starts[: min(len(starts), 250_000)]:
        stack=[(start_key, [])]; visited={start_key}
        while stack and len(out)<n:
            current,path=stack.pop()
            if len(path)>=max_hops: continue
            for edge in by_subject.get(current, ()):
                np=path+[list(edge)]; pnames=[x[1] for x in np]
                if len(np)>=2:
                    inferred=None
                    if len(np)==2: inferred=safe.get((pnames[0],pnames[1]))
                    elif len(np)==3:
                        a=safe.get((pnames[0],pnames[1])); inferred=safe.get((a,pnames[2])) if a else None
                    if inferred:
                        s=np[0][0]; o=np[-1][2]; key=(s.lower(),inferred,o.lower())
                        if key not in seen and s.lower()!=o.lower() and o not in GENERIC_WORDS:
                            seen.add(key)
                            out.append({"question":natural_question(inferred,s,o),"status":"SUPPORTED","kind":"indirect","subject":s,"predicate":inferred,"object":o,"hops":len(np),"path":np})
                obj=edge[2].lower()
                if obj not in visited and obj in by_subject:
                    visited.add(obj); stack.append((obj,np))
    return out

def sample_unknown(con, adapter, n, seed):
    predicates = [p for p in POSITIVE_FAMILIES if p in {"is_a","has_part","has","has_property","capable_of","part_of","made_of","located_in"}]
    rows = fetch_rows(con, adapter, predicates, max(50_000,n*500))
    rng=random.Random(seed); rng.shuffle(rows)
    # One bulk exact-positive set and a compact contrast pool per predicate.
    positives={(str(r["subject"]).lower(),str(r["predicate"]),str(r["object_text"]).lower()) for r in rows}
    pools=defaultdict(list)
    for r in rows:
        pools[str(r["predicate"])].append(str(r["object_text"]))
    out=[]; seen=set()
    for r in rows:
        s=str(r["subject"]); p=str(r["predicate"]); original=str(r["object_text"])
        pool=pools[p]
        for alt in rng.sample(pool, min(20,len(pool))):
            key=(s.lower(),p,alt.lower())
            if alt.lower()==original.lower() or key in positives or key in seen: continue
            seen.add(key)
            out.append({"question":natural_question(p,s,alt),"status":"UNKNOWN","kind":"hard_unknown","subject":s,"predicate":p,"object":alt,"contrast_object":original})
            break
        if len(out)>=n: break
    return out

def summarize_goals(rels):
    c=Counter()
    for r in rels:
        fam=r["family"]
        if fam!="other": c[fam]+=1
    return dict(c)

def run(args):
    started=time.perf_counter(); con=connect(args.memory); info=inspect_schema(con); adapter,candidates=pick_adapter(con,info,args.table)
    print("=== V542 DB-NATIVE KNOWLEDGE VERDICT DISCOVERY ===")
    print(f"memory       : {args.memory}"); print(f"knowledge    : {adapter.table} ({adapter.mode})"); print(f"workers      : {args.workers}"); print("LLM          : NOT USED"); print("conversation  : NOT USED")
    print("\n=== KNOWLEDGE TABLE CANDIDATES ===")
    for c in sorted(candidates,key=lambda x:x["score"],reverse=True): print(f"  {c['table']:<28} mode={c['mode']:<9} rows={c['rows']:<10d} excluded={c['excluded']}")
    rels=relation_inventory(con,adapter)
    print("\n=== ACTUAL RELATION INVENTORY ===")
    for r in rels[:80]: print(f"  {r['predicate']:<28} {r['count']:>10}  family={r['family']}")
    t=time.perf_counter(); direct=sample_direct(con,adapter,args.direct,args.seed); print(f"\n[DISCOVERY] direct supported={len(direct)} seconds={time.perf_counter()-t:.3f}")
    t=time.perf_counter(); indirect=mine_indirect(con,adapter,args.indirect,args.max_hops,args.seed); print(f"[DISCOVERY] indirect supported={len(indirect)} seconds={time.perf_counter()-t:.3f}")
    t=time.perf_counter(); neg=sample_negative(con,adapter,args.negative,args.seed+1); print(f"[DISCOVERY] explicit negative={len(neg)} seconds={time.perf_counter()-t:.3f}")
    t=time.perf_counter(); unk=sample_unknown(con,adapter,args.unknown,args.seed+2); print(f"[DISCOVERY] hard unknown={len(unk)} seconds={time.perf_counter()-t:.3f}")
    print("\n=== GOALS / QUERY FAMILIES OBSERVED IN DB ===")
    for k,v in summarize_goals(rels).items(): print(f"  {k:<22} predicates={v}")
    print("\n=== DISTRIBUTION ===")
    total=len(direct)+len(indirect)+len(neg)+len(unk)
    for label,count in (("DIRECT-SUPPORTED",len(direct)),("INDIRECT-SUPPORTED",len(indirect)),("REFUTED",len(neg)),("UNKNOWN",len(unk))):
        pct=(100*count/total) if total else 0; print(f"{label:<20}: {count:>6} ({pct:5.1f}%)")
    def show(title,rows):
        print(f"\n=== {title} ===")
        for r in rows[:args.show]:
            if title=="SAMPLE UNKNOWN": print(f"  {r['question']} -> UNKNOWN (contrast={r.get('contrast_object')})")
            else: print(f"  {r['question']} -> {r['status']} via {r.get('path')}")
        if not rows: print("  <none>")
    show("SAMPLE DIRECT",direct); show("SAMPLE INDIRECT",indirect); show("SAMPLE REFUTED",neg); show("SAMPLE UNKNOWN",unk)
    report={"benchmark":"v542_db_native_knowledge_verdict_discovery","selected_knowledge_table":{"name":adapter.table,"mode":adapter.mode},"table_candidates":candidates,"relation_inventory":rels,"goal_families":summarize_goals(rels),"direct_supported":direct,"indirect_supported":indirect,"explicit_negative":neg,"unknown":unk,"workers":args.workers,"elapsed_seconds":time.perf_counter()-started}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)),exist_ok=True)
        with open(args.json,"w",encoding="utf-8") as f: json.dump(report,f,indent=2)
        print("\nJSON written:",args.json)

def main():
    ap=argparse.ArgumentParser(description="Discover supported, indirect, explicitly refuted, and unknown claims from the real semantic graph")
    ap.add_argument("--memory",required=True); ap.add_argument("--table"); ap.add_argument("--direct",type=int,default=500); ap.add_argument("--indirect",type=int,default=500); ap.add_argument("--unknown",type=int,default=500); ap.add_argument("--negative",type=int,default=100); ap.add_argument("--max-hops",type=int,default=3); ap.add_argument("--workers",type=int,default=DEFAULT_WORKERS); ap.add_argument("--seed",type=int,default=542); ap.add_argument("--show",type=int,default=10); ap.add_argument("--json")
    a=ap.parse_args(); a.workers=max(1,min(64,a.workers)); a.max_hops=max(1,min(4,a.max_hops)); run(a)
if __name__=="__main__": main()
