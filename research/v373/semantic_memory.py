
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import re, sqlite3, math


def canonical_concept(value: str) -> str:
    x=str(value).strip().lower()
    x=re.sub(r"^/c/[a-z]{2}/","",x)
    x=x.split("/")[0]
    x=x.replace("_"," ").replace("-"," ")
    x=re.sub(r"[^a-z0-9' ]+"," ",x)
    x=re.sub(r"\s+"," ",x).strip()
    return x


@dataclass(frozen=True)
class SemanticEdge:
    source:str
    relation:str
    target:str
    weight:float=1.0
    provenance:str="conceptnet"


@dataclass(frozen=True)
class SemanticCandidate:
    concept:str
    prior:float
    evidence:Tuple[SemanticEdge,...]=()
    score:float=0.0
    confidence:float=0.0


@dataclass(frozen=True)
class GroundingBelief:
    query:str
    candidates:Tuple[SemanticCandidate,...]
    committed:Optional[str]
    confidence:float
    entropy:float
    evidence_count:int
    revision:int


class IndexedSemanticMemory:
    """
    ConceptNet-backed semantic memory.

    Real DB mode builds a normalized concept index once. Synthetic mode is used
    by the smoke test. Queries then touch only the indexed concept bucket,
    rather than scanning the full edge table per grounding request.
    """

    def __init__(self):
        self.adj:Dict[str,List[SemanticEdge]]=defaultdict(list)
        self.reverse:Dict[str,List[SemanticEdge]]=defaultdict(list)
        self.edge_count=0
        self.source="synthetic"

    @classmethod
    def from_edges(cls, edges:Iterable[SemanticEdge]):
        mem=cls()
        for edge in edges:
            mem.add(edge)
        return mem

    def add(self,edge:SemanticEdge):
        s=canonical_concept(edge.source)
        t=canonical_concept(edge.target)
        if not s or not t:
            return
        norm=SemanticEdge(
            source=s,
            relation=str(edge.relation),
            target=t,
            weight=float(edge.weight),
            provenance=edge.provenance,
        )
        self.adj[s].append(norm)
        self.reverse[t].append(norm)
        self.edge_count+=1

    def concepts(self):
        return set(self.adj)|set(self.reverse)

    def neighborhood(self, concept:str, relation:Optional[str]=None, max_edges:int=32):
        key=canonical_concept(concept)
        edges=[]
        for e in self.adj.get(key,()):
            if relation is None or e.relation==relation:
                edges.append(e)
        for e in self.reverse.get(key,()):
            if relation is None or e.relation==relation:
                edges.append(
                    SemanticEdge(
                        source=e.target,
                        relation=e.relation,
                        target=e.source,
                        weight=e.weight,
                        provenance=e.provenance,
                    )
                )
        return tuple(edges[:max_edges])

    def relation_exists(self, source:str, relation:str, target:str):
        s=canonical_concept(source); t=canonical_concept(target)
        return any(
            e.target==t and e.relation==relation
            for e in self.adj.get(s,())
        )

    def retrieve(self, query:str, max_candidates:int=8, hop:int=1):
        q=canonical_concept(query)
        if not q:
            return ()

        # Exact concept hits first. Multi-concept surface strings can still
        # map to multiple candidates in real databases.
        seeds=[]
        if q in self.concepts():
            seeds.append(q)

        # Token-composed candidates.
        for token in q.split():
            if token in self.concepts() and token not in seeds:
                seeds.append(token)

        out=[]
        seen=set()
        for seed in seeds:
            score=1.0 if seed==q else 0.6
            candidate=SemanticCandidate(
                concept=seed,
                prior=score,
                evidence=self.neighborhood(seed,max_edges=16),
            )
            if seed not in seen:
                out.append(candidate); seen.add(seed)

            if len(out)>=max_candidates:
                break

        return tuple(out)


class ConceptNetSQLiteLoader:
    """
    Load the compact ConceptNet DB into IndexedSemanticMemory. Supports the
    same common column layouts as the earlier loader, but constructs an index
    in one pass.
    """

    def __init__(self,db_path:Path):
        self.db_path=Path(db_path)
        self.conn=None
        self.table=None
        self.columns=()

    def connect(self):
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        self.conn=sqlite3.connect(
            f"file:{self.db_path.resolve()}?mode=ro",
            uri=True,
        )
        self._discover()

    def _discover(self):
        tables=[
            r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols=[
                r[1] for r in self.conn.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]
            low={c.lower() for c in cols}
            if (
                any(x in low for x in ("source","src","start","subject","from_id"))
                and any(x in low for x in ("target","dst","end","object","to_id"))
                and any(x in low for x in ("relation","rel","predicate","relation_id"))
            ):
                self.table=table
                self.columns=tuple(cols)
                return
        raise RuntimeError(f"No edge table found: {tables}")

    def _col(self,names):
        low={c.lower():c for c in self.columns}
        for n in names:
            if n in low: return low[n]
        raise RuntimeError(f"Missing {names}")

    def load_index(self):
        self.connect()
        src=self._col(("source","src","start","subject","from_id"))
        dst=self._col(("target","dst","end","object","to_id"))
        rel=self._col(("relation","rel","predicate","relation_id"))

        memory=IndexedSemanticMemory()
        memory.source=str(self.db_path.resolve())

        for row in self.conn.execute(
            f'SELECT "{src}","{rel}","{dst}" FROM "{self.table}"'
        ):
            memory.add(
                SemanticEdge(
                    source=str(row[0]),
                    relation=str(row[1]),
                    target=str(row[2]),
                )
            )
        return memory

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn=None


@dataclass
class SemanticHypothesis:
    concept:str
    prior:float
    support:float=0.0
    contradiction:float=0.0

    @property
    def posterior_log_score(self):
        return (
            math.log(max(1e-12,self.prior))
            +0.9*self.support
            -1.5*self.contradiction
        )


class SemanticBeliefManager:
    """
    Maintains multiple competing interpretations for each grounding query.

    Evidence is append-only and scoped to the query. It can be revised without
    mutating the underlying semantic memory.
    """

    def __init__(self,commit_threshold=0.80,commit_margin=0.20):
        self.commit_threshold=commit_threshold
        self.commit_margin=commit_margin
        self.beliefs:Dict[str,GroundingBelief]={}
        self.evidence:List[Tuple[str,str,bool,str]]= []

    def candidates(self, query, memory):
        retrieved=memory.retrieve(query,max_candidates=8)
        return [
            SemanticHypothesis(
                concept=x.concept,
                prior=max(1e-3,x.prior),
            )
            for x in retrieved
        ]

    def update(self, query, candidates, evidence_items, revision=1):
        scored=[]
        for c in candidates:
            support=sum(
                1 for item in evidence_items
                if item[0]==c.concept and item[1]
            )
            contradiction=sum(
                1 for item in evidence_items
                if item[0]==c.concept and not item[1]
            )
            c=SemanticHypothesis(
                c.concept,
                c.prior,
                support,
                contradiction,
            )

            # Hard contradiction penalty: one verified counterexample must
            # materially reduce confidence instead of being averaged away.
            log_score=(
                math.log(max(1e-12,c.prior))
                +1.2*support
                -2.5*contradiction
            )
            scored.append((c,log_score))

        if not scored:
            belief=GroundingBelief(
                query,(),None,0.0,0.0,
                len(evidence_items),
                revision,
            )
            self.beliefs[query]=belief
            return belief

        mx=max(x[1] for x in scored)
        weights=[
            math.exp(x[1]-mx)
            for x in scored
        ]
        z=sum(weights) or 1.0
        ranked=sorted(
            zip(scored,weights),
            key=lambda x:x[1],
            reverse=True,
        )
        posterior=[
            (pair[0],weight/z)
            for pair,weight in ranked
        ]

        entropy=-sum(
            p*math.log(max(p,1e-12))
            for _,p in posterior
        )
        best=posterior[0]
        second=posterior[1] if len(posterior)>1 else None
        confidence=best[1]
        margin=confidence-(second[1] if second else 0.0)

        committed=(
            best[0].concept
            if confidence>=self.commit_threshold
            and margin>=self.commit_margin
            and best[0].contradiction==0
            else None
        )

        out=[]
        for c,p in posterior:
            out.append(
                SemanticCandidate(
                    concept=c.concept,
                    prior=c.prior,
                    evidence=(),
                    score=c.posterior_log_score,
                    confidence=p,
                )
            )

        belief=GroundingBelief(
            query=query,
            candidates=tuple(out),
            committed=committed,
            confidence=confidence,
            entropy=entropy,
            evidence_count=len(evidence_items),
            revision=revision,
        )
        self.beliefs[query]=belief
        return belief

    def record(self,query,concept,success,reason):
        self.evidence.append(
            (concept,bool(success),reason,query)
        )

    def evidence_for(self,query):
        return [
            (concept,success,reason)
            for concept,success,reason,q
            in self.evidence
            if q==query
        ]


def memory_evidence(concept, evidence_items):
    return ()


class SemanticGroundingController:
    """
    Cognitive-layer grounding loop.

    It chooses between retrieve, compare, test and commit. In a real system,
    test() will be supplied by the cognitive environment. Smoke uses explicit
    graph constraints as the environment.
    """

    def __init__(self, memory:IndexedSemanticMemory):
        self.memory=memory
        self.beliefs=SemanticBeliefManager()

    def retrieve(self,query):
        return self.beliefs.candidates(
            query,
            self.memory,
        )

    def semantic_consistency(self, concept, context):
        """
        Score a candidate against an explicit semantic context:
        context is a list of (relation, target) constraints.
        """
        if not context:
            return 0.5
        matches=0
        for relation,target in context:
            if self.memory.relation_exists(
                concept,
                relation,
                target,
            ):
                matches+=1
        return matches/max(1,len(context))

    def ground(
        self,
        query,
        context=(),
        intervention=None,
        revision=1,
    ):
        candidates=self.retrieve(query)
        if not candidates:
            return None

        scored=[]
        current_evidence=[]

        for c in candidates:
            consistency=self.semantic_consistency(
                c.concept,
                context,
            )
            scored.append(
                SemanticHypothesis(
                    concept=c.concept,
                    prior=c.prior,
                    support=consistency,
                    contradiction=(1.0-consistency),
                )
            )
            current_evidence.append(
                (
                    c.concept,
                    consistency>=0.5,
                    "graph_consistency",
                )
            )

        # Query-context evidence belongs to this revision. Persistent semantic
        # memory is retained separately; stale interpretation evidence must not
        # make a later context unable to revise the belief.
        for concept,success,reason in current_evidence:
            self.beliefs.record(
                f"{query}#rev{revision}",
                concept,
                success,
                reason,
            )

        belief=self.beliefs.update(
            query,
            scored,
            current_evidence,
            revision=revision,
        )

        return belief

