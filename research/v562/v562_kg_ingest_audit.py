from __future__ import annotations
import argparse, bz2, json, os, re, sqlite3, time, zipfile
from pathlib import Path

YAGO_FILES = {
    "facts": "yago-4.6-facts.zip",
    "taxonomy": "yago-4.6-taxonomy.zip",
    "schema": "yago-4.6-schema.zip",
    "beyond": "yago-4.6-beyond-wikipedia.zip",
    "labels": "yago-4.6-labels.zip",
    "beyond_labels": "yago-4.6-beyond-wikipedia-labels.zip",
}
DBPEDIA_FILES = {
    "objects": "mappingbased-objects_lang=en.ttl.bz2",
    "ontology": "ontology--DEV_type=parsed_sorted.nt",
}
PREFIXES = {
    "rdf:type":"is_a", "rdfs:subClassOf":"is_a", "rdfs:subclassOf":"is_a",
    "schema:subClassOf":"is_a", "schema:type":"is_a", "type":"is_a", "subclass_of":"is_a",
}

def clean(v):
    v = v.strip()
    if v.startswith("<") and v.endswith(">"): v = v[1:-1]
    return v

def rel(v):
    v = clean(v)
    if "#" in v: v = v.rsplit("#",1)[-1]
    if "/" in v: v = v.rstrip("/").rsplit("/",1)[-1]
    v = PREFIXES.get(v, v)
    return {
        "isA":"is_a", "is_a":"is_a", "subclass":"is_a",
        "capableOf":"capable_of", "usedFor":"used_for", "partOf":"part_of",
        "hasPart":"has_part", "locatedIn":"located_in", "relatedTo":"related_to",
        "hasProperty":"has_property"
    }.get(v, v)

def ent(v):
    v = clean(v)
    return "" if v.startswith('"') else v

def open_db(path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): p.unlink()
    for suffix in ("-wal", "-shm"):
        q = Path(str(p)+suffix)
        if q.exists(): q.unlink()
    c = sqlite3.connect(str(p))
    # Critical fix: do NOT use WAL + huge in-memory cache for this bulk loader.
    c.execute("PRAGMA journal_mode=DELETE")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA temp_store=FILE")
    c.execute("PRAGMA cache_size=-32768")  # ~32 MiB page cache
    c.execute("PRAGMA mmap_size=0")
    c.execute("PRAGMA locking_mode=NORMAL")
    c.executescript("""
      CREATE TABLE edges(subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL, source TEXT NOT NULL);
      CREATE TABLE sources(source TEXT PRIMARY KEY, triples INTEGER);
      CREATE TABLE stats(key TEXT PRIMARY KEY, value TEXT);
    """)
    c.commit()
    return c

def zip_member(z):
    ms = [x for x in z.namelist() if not x.endswith("/")]
    if not ms: raise RuntimeError("ZIP contains no files")
    # Prefer actual RDF/TSV payload, not metadata.
    for ext in (".ttl", ".nt", ".tsv"):
        hits = [m for m in ms if m.lower().endswith(ext)]
        if hits: return hits[0]
    return ms[0]

def yago_line(line):
    if not line.strip() or line.lstrip().startswith("#"): return None
    p = line.rstrip("\n").split("\t")
    if len(p) < 3: return None
    if len(p) >= 4 and p[0].startswith("<") and (p[1].startswith("<") or ":" in p[1]):
        s, pred, o = p[1], p[2], p[3]
    else:
        s, pred, o = p[:3]
    s, pred, o = ent(s), rel(pred), ent(o.rstrip(" ."))
    return (s,pred,o) if s and pred and o else None

# DBpedia mappingbased-objects is N-Triples/Turtle-like RDF.  Do not use a
# whitespace regex for the whole line: URI/literal escaping and very long
# objects make that brittle.  Parse the first two RDF terms explicitly and
# then treat the remainder as the object term.
TERM_RE = re.compile(r"^\s*(<[^>]*>|(?:[A-Za-z_][\w.-]*):[^\s]+)\s+(<[^>]*>|(?:[A-Za-z_][\w.-]*):[^\s]+)\s+(.*)$")

def rdf_line(line):
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('@prefix') or line.startswith('PREFIX '):
        return None
    # N-Triples/Turtle statements terminate in a dot.  Keep dots inside URIs.
    if not line.endswith('.'):
        return None
    body = line[:-1].rstrip()
    m = TERM_RE.match(body)
    if not m:
        return None
    s, p, o = m.groups()
    s, p = ent(s), rel(p)
    o = o.strip()
    # We only ingest entity/entity edges. Literals are deliberately excluded.
    if o.startswith('"') or o.startswith("'"):
        return None
    # Remove optional RDF-star wrapping and punctuation conservatively.
    if o.endswith(' .'):
        o = o[:-2].rstrip()
    o = ent(o)
    return (s, p, o) if s and p and o else None

def insert_batch(c, batch):
    if batch:
        c.executemany("INSERT INTO edges VALUES (?,?,?,?)", batch)
        c.commit()
        batch.clear()

def ingest_yago(c, path, name, batch_size, progress_lines, max_lines=None):
    print(f"[YAGO] {name}: {path}", flush=True)
    if not path.exists():
        print("  MISSING -> skipped", flush=True); return 0
    total=n=0; last_report=0; started=time.time()
    with zipfile.ZipFile(path) as z:
        m=zip_member(z); print("  member:",m, flush=True)
        with z.open(m) as f:
            batch=[]
            for raw in f:
                if max_lines is not None and total >= max_lines:
                    break
                total += 1
                t=yago_line(raw.decode("utf-8","replace"))
                if t:
                    batch.append((*t,name)); n += 1
                    if len(batch) >= batch_size:
                        c.executemany("INSERT INTO edges VALUES (?,?,?,?)", batch); c.commit(); batch.clear()
                if total-last_report >= progress_lines:
                    elapsed=max(time.time()-started,0.001)
                    print(f"  progress lines={total:,} triples={n:,} rate={total/elapsed:,.0f}/s", flush=True)
                    last_report=total
            insert_batch(c,batch)
    c.execute("INSERT OR REPLACE INTO sources VALUES (?,?)",(name,n)); c.commit()
    print(f"  DONE lines={total:,} triples={n:,} elapsed={time.time()-started:.1f}s", flush=True)
    return n

def ingest_bz2(c, path, name, batch_size, progress_lines, max_lines=None):
    print(f"[DBPEDIA] {name}: {path}", flush=True)
    if not path.exists():
        print("  MISSING -> skipped", flush=True); return 0
    total = accepted = rejected = literals = 0
    last_report = 0
    started = time.time()
    opener = bz2.open if path.suffix == ".bz2" else open
    kwargs = {"mode":"rt", "encoding":"utf-8", "errors":"replace"} if path.suffix == ".bz2" else {"mode":"r", "encoding":"utf-8", "errors":"replace"}
    reject_samples = []
    with opener(path, **kwargs) as f:
        batch=[]
        for line in f:
            if max_lines is not None and total >= max_lines:
                break
            total += 1
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                rejected += 1
            else:
                t = rdf_line(line)
                if t:
                    batch.append((*t,name)); accepted += 1
                    if len(batch) >= batch_size:
                        c.executemany("INSERT INTO edges VALUES (?,?,?,?)", batch); c.commit(); batch.clear()
                else:
                    if '"' in stripped and not stripped.startswith('"'):
                        literals += 1
                    else:
                        rejected += 1
                        if len(reject_samples) < 3:
                            reject_samples.append(stripped[:500])
            if total-last_report >= progress_lines:
                elapsed=max(time.time()-started,0.001)
                print(f"  progress lines={total:,} triples={accepted:,} rejected={rejected:,} literals={literals:,} rate={total/elapsed:,.0f}/s", flush=True)
                last_report=total
        insert_batch(c,batch)
    c.execute("INSERT OR REPLACE INTO sources VALUES (?,?)",(name,accepted)); c.commit()
    elapsed=time.time()-started
    print(f"  DONE lines={total:,} triples={accepted:,} rejected={rejected:,} literals={literals:,} elapsed={elapsed:.1f}s", flush=True)
    if reject_samples:
        print("  reject samples:", flush=True)
        for sample in reject_samples:
            print("    "+sample, flush=True)
    return accepted

def build_indexes(c):
    print("\n[INDEX] building indexes after bulk load (much cheaper than maintaining them per row)", flush=True)
    for sql in (
        "CREATE INDEX idx_s ON edges(subject)",
        "CREATE INDEX idx_r ON edges(relation)",
        "CREATE INDEX idx_o ON edges(object)",
        "CREATE INDEX idx_sro ON edges(subject,relation,object)",
    ):
        c.execute(sql); c.commit()
    c.execute("ANALYZE"); c.commit()

def stats(c):
    g={
      "edges":c.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
      "concepts":c.execute("SELECT COUNT(*) FROM (SELECT subject x FROM edges UNION SELECT object FROM edges)").fetchone()[0],
      "subjects":c.execute("SELECT COUNT(DISTINCT subject) FROM edges").fetchone()[0],
      "relations":c.execute("SELECT COUNT(DISTINCT relation) FROM edges").fetchone()[0],
    }
    print("\n=== GRAPH ===")
    for k,v in g.items(): print(f"{k:12s}{v:,}")
    print("\n=== SOURCES ===")
    for r in c.execute("SELECT source,triples FROM sources ORDER BY source"):
        print(f"{r[0]:28s}{r[1]:,}")
    return g

def composition(c,min_paths):
    print("\n=== COMPOSITION AUDIT ===", flush=True)
    pairs=c.execute("""
      SELECT e1.relation,e2.relation,COUNT(*)
      FROM edges e1 JOIN edges e2 ON e1.object=e2.subject
      WHERE e1.subject!=e2.object
      GROUP BY e1.relation,e2.relation HAVING COUNT(*)>=?
      ORDER BY COUNT(*) DESC
    """,(min_paths,)).fetchall()
    print("candidate relation pairs:",len(pairs), flush=True)
    out=[]
    for r1,r2,paths in pairs:
        confirm=c.execute("""
          SELECT COUNT(*) FROM (
            SELECT DISTINCT e1.subject s,e2.object o
            FROM edges e1 JOIN edges e2 ON e1.object=e2.subject
            WHERE e1.relation=? AND e2.relation=? AND e1.subject!=e2.object
          ) p JOIN edges d ON d.subject=p.s AND d.object=p.o
        """,(r1,r2)).fetchone()[0]
        out.append({"sequence":[r1,r2],"paths":paths,"direct_confirmations":confirm,
                    "precision":confirm/paths if paths else 0})
    out.sort(key=lambda x:(x["precision"],x["direct_confirmations"],x["paths"]),reverse=True)
    for x in out[:50]:
        print(f"{' -> '.join(x['sequence']):42s} paths={x['paths']:8,} confirm={x['direct_confirmations']:7,} precision={x['precision']:.5f}")
    return out

def audit(c,args):
    g=stats(c)
    div=c.execute("SELECT COUNT(*) FROM (SELECT relation FROM edges GROUP BY relation HAVING COUNT(*)>=100)").fetchone()[0]
    overlap=c.execute("""
      SELECT COUNT(*) FROM
      (SELECT DISTINCT subject FROM edges WHERE source LIKE 'yago%') y
      JOIN
      (SELECT DISTINCT subject FROM edges WHERE source LIKE 'dbpedia%') d
      ON y.subject=d.subject
    """).fetchone()[0]
    comps=composition(c,args.min_paths)
    strong=[x for x in comps if x["paths"]>=1000 and x["direct_confirmations"]>=100 and x["precision"]>=.02]
    usable=[x for x in comps if x["paths"]>=250 and x["direct_confirmations"]>=25 and x["precision"]>=.01]
    ss=sum(x["direct_confirmations"] for x in strong); us=sum(x["direct_confirmations"] for x in usable)
    if len(strong)>=10 and ss>=5000 and div>=20: verdict="STRONG"
    elif len(strong)>=3 and (ss>=500 or us>=1000): verdict="USABLE"
    elif len(usable)>=5 and us>=200: verdict="WEAK"
    else: verdict="INSUFFICIENT"
    v={"label":verdict,"strong_rules":len(strong),"usable_rules":len(usable),
       "strong_confirmation_support":ss,"usable_confirmation_support":us,
       "max_rule_precision":max((x["precision"] for x in comps),default=0),
       "composition_candidates":len(comps),"relations_ge_100":div,
       "shared_subject_identifiers":overlap}
    print("\n"+"="*72+"\nDATA VERDICT\n"+"="*72)
    for k,val in v.items(): print(f"{k:28s}: {val}")
    return {"benchmark":"v561_knowledge_graph_ingestion_audit_fixed","graph":g,"verdict":v,
            "composition":{"candidates":len(comps),"top_rules":comps[:100]}}

def main():
    p=argparse.ArgumentParser(description="v562 KG ingestion + semantic composition audit")
    p.add_argument("--yago",default=r".\\data\\yago")
    p.add_argument("--dbpedia",default=r".\\data\\dbpedia")
    p.add_argument("--output",default=r".\\results\\v562_kg_composition_audit.sqlite")
    p.add_argument("--json",default=r".\\results\\v562_kg_composition_audit.json")
    p.add_argument("--min-paths",type=int,default=25)
    p.add_argument("--batch-size",type=int,default=25000)
    p.add_argument("--progress-lines",type=int,default=1000000)
    p.add_argument("--yago-set",default="core",choices=["core","all"])
    p.add_argument("--dbpedia-only",action="store_true",help="Skip YAGO entirely")
    p.add_argument("--max-lines",type=int,default=None,help="Maximum input lines per source (diagnostic/smoke mode)")
    p.add_argument("--smoke-test",action="store_true",help="Convenience mode: DBpedia-only, 5M lines/source, lower composition threshold")
    a=p.parse_args()
    if a.smoke_test:
        a.dbpedia_only=True
        if a.max_lines is None: a.max_lines=5_000_000
        if a.min_paths==25: a.min_paths=5
    t=time.time()
    print("v562 KG INGEST + COMPOSITION AUDIT", flush=True)
    print(f"  dbpedia_only={a.dbpedia_only} max_lines={a.max_lines} yago_set={a.yago_set}", flush=True)
    c=open_db(a.output); yd,dd=Path(a.yago),Path(a.dbpedia)
    if not a.dbpedia_only:
        names=list(YAGO_FILES.items()) if a.yago_set=="all" else [(k,YAGO_FILES[k]) for k in ("facts","taxonomy","schema","beyond")]
        for k,f in names:
            ingest_yago(c,yd/f,"yago_"+k,a.batch_size,a.progress_lines,a.max_lines)
    for k,f in DBPEDIA_FILES.items():
        ingest_bz2(c,dd/f,"dbpedia_"+k,a.batch_size,a.progress_lines,a.max_lines)
    build_indexes(c)
    result=audit(c,a)
    result["benchmark"]="v562_knowledge_graph_ingestion_audit"
    result["config"]={
        "dbpedia_only":a.dbpedia_only,"max_lines_per_source":a.max_lines,
        "yago_set":a.yago_set,"min_paths":a.min_paths,
        "batch_size":a.batch_size,"progress_lines":a.progress_lines,
    }
    result["elapsed_seconds"]=time.time()-t
    Path(a.json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.json).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    c.close()
    print(f"\nSQLite: {a.output}\nJSON:   {a.json}\nElapsed: {result['elapsed_seconds']:.1f}s", flush=True)

if __name__=="__main__": main()
