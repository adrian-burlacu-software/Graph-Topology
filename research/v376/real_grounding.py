
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sqlite3,re,math

def canonical(x):
    x=str(x).strip().lower()
    x=re.sub(r"^/c/[a-z]{2}/","",x)
    x=x.split("/")[0].replace("_"," ").replace("-"," ")
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9' ]+"," ",x)).strip()

@dataclass(frozen=True)
class Edge:
    source:str
    relation:str
    target:str
    weight:float=1.0
    provenance:str="conceptnet"

class IndexedConceptNet:
    def __init__(self,db_path):
        self.db_path=Path(db_path)
        self.conn=None
        self.table=None
        self.columns=()
        self.adj=defaultdict(list)
        self.rev=defaultdict(list)
        self.edge_count=0
        self.concepts_set=set()

    def connect(self):
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        self.conn=sqlite3.connect(
            f"file:{self.db_path.resolve()}?mode=ro",
            uri=True,
        )
        tables=[r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )]
        for t in tables:
            cols=[r[1] for r in self.conn.execute(f'PRAGMA table_info("{t}")')]
            low={c.lower() for c in cols}
            if (
                any(x in low for x in ("source","src","start","subject","from_id"))
                and any(x in low for x in ("target","dst","end","object","to_id"))
                and any(x in low for x in ("relation","rel","predicate","relation_id"))
            ):
                self.table=t
                self.columns=tuple(cols)
                break
        if not self.table:
            raise RuntimeError(f"No ConceptNet edge table found: {tables}")

    def col(self,names):
        low={c.lower():c for c in self.columns}
        for n in names:
            if n in low: return low[n]
        raise RuntimeError(f"Missing column among {names}")

    def build_index(self):
        self.connect()
        src=self.col(("source","src","start","subject","from_id"))
        dst=self.col(("target","dst","end","object","to_id"))
        rel=self.col(("relation","rel","predicate","relation_id"))
        for s,r,t in self.conn.execute(
            f'SELECT "{src}","{rel}","{dst}" FROM "{self.table}"'
        ):
            ss,tt=canonical(s),canonical(t)
            if not ss or not tt:
                continue
            e=Edge(ss,str(r),tt)
            self.adj[ss].append(e)
            self.rev[tt].append(e)
            self.concepts_set.add(ss); self.concepts_set.add(tt)
            self.edge_count+=1
        return self

    @property
    def concepts(self):
        return self.concepts_set

    def neighborhood(self,c,max_edges=64):
        c=canonical(c)
        out=[]
        seen=set()
        for e in self.adj.get(c,()):
            k=(e.relation,e.target)
            if k not in seen:
                seen.add(k); out.append(e)
        for e in self.rev.get(c,()):
            k=(e.relation,e.source)
            if k not in seen:
                seen.add(k)
                out.append(Edge(c,e.relation,e.source))
        return out[:max_edges]

    def relation_exists(self,s,r,t):
        s,t=canonical(s),canonical(t)
        return any(e.target==t and e.relation==r for e in self.adj.get(s,()))

    def surface_candidates(self,surface,max_candidates=6):
        q=canonical(surface)
        exact=[q] if q in self.concepts else []
        multi=[
            c for c in self.concepts
            if c.startswith(q+" ")
        ]
        # Prefer exact sense + phrase senses; then any concept sharing first word.
        vals=[]
        for c in exact+multi:
            if c not in vals: vals.append(c)
        if len(vals)>=max_candidates:
            return vals[:max_candidates]
        for c in self.concepts:
            if c.startswith(q+" ") and c not in vals:
                vals.append(c)
                if len(vals)>=max_candidates: break
        return vals[:max_candidates]

    def find_ambiguous_surfaces(self, limit=25):
        """
        Finds real surface forms with >=2 candidate concepts and divergent
        neighborhoods. Prefix-based ambiguity is explicit and deterministic.
        """
        buckets=defaultdict(list)
        for c in self.concepts:
            if not c or len(c)<3: continue
            surface=c.split(" ")[0]
            buckets[surface].append(c)

        scored=[]
        for s,cs in buckets.items():
            if len(cs)<2: continue
            cs=sorted(cs,key=lambda x:len(self.adj.get(x,()))+len(self.rev.get(x,())),reverse=True)[:8]
            for i,a in enumerate(cs):
                ea={(e.relation,e.target) for e in self.adj.get(a,())}
                if not ea: continue
                for b in cs[i+1:]:
                    eb={(e.relation,e.target) for e in self.adj.get(b,())}
                    div=len(ea.symmetric_difference(eb))
                    if div<2: continue
                    score=div+min(len(ea),len(eb))
                    scored.append((score,s,a,b))
        scored.sort(reverse=True)
        return [(s,a,b,score) for score,s,a,b in scored[:limit]]

    def close(self):
        if self.conn: self.conn.close(); self.conn=None
