
from __future__ import annotations

import argparse, json, re, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from babylm_grammar import BabyLMReader, GrammarCognitiveLearner
from semantic_memory import (
    IndexedSemanticMemory, SemanticEdge, canonical_concept,
)
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


TOKEN_RE=re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+")

DET={"the","a","an"}
AUX={
    "is","are","was","were","am","be","been","being",
    "do","does","did","have","has","had",
    "can","could","will","would","should","may","might","must",
}
VERBS={
    "chase","chases","eat","eats","see","sees","like","likes",
    "want","wants","make","makes","take","takes","used","use",
    "need","needs","comes","come","lose","loses",
}


@dataclass(frozen=True)
class Frame:
    predicate:str
    agent:str
    patient:str

    def normalized(self):
        return Frame(
            canonical_concept(self.predicate),
            canonical_concept(self.agent),
            canonical_concept(self.patient),
        )


@dataclass(frozen=True)
class Perceived:
    sentence:str
    frame:Optional[Frame]
    grammar_rule:Optional[str]
    confidence:float
    lexical_modes:Tuple[str,...]


def lemma_candidates(token):
    t=canonical_concept(token)
    out=[t]
    if t.endswith("ies") and len(t)>4: out.append(t[:-3]+"y")
    if t.endswith("es") and len(t)>4: out.append(t[:-2])
    if t.endswith("s") and len(t)>3: out.append(t[:-1])
    if t.endswith("ed") and len(t)>4:
        out.extend((t[:-2],t[:-1]))
    if t.endswith("ing") and len(t)>5: out.append(t[:-3])
    seen=set(); ans=[]
    for x in out:
        if x and x not in seen:
            seen.add(x); ans.append(x)
    return tuple(ans)


class GraphLexicon:
    def __init__(self,memory):
        self.memory=memory
        self.concepts=memory.concepts()

    def resolve(self,token):
        for c in lemma_candidates(token):
            if c in self.concepts:
                return c,("exact" if c==canonical_concept(token) else "morphological")
        return None,"unresolved"


class RealRoundtrip:
    def __init__(self,learner,lexicon):
        self.learner=learner
        self.semantic=learner.semantic
        self.lexicon=lexicon
        self.learned_svo=any(
            r.split("::",1)[0]=="DET|NOUN|VERB"
            for r in learner.grammar.memory.rules
        )

    def tag(self,t):
        if t in DET: return "DET"
        if t in AUX: return "AUX"
        if t in VERBS: return "VERB"
        if t.endswith("ing") or t.endswith("ed"): return "VERB"
        return "NOUN"

    def perceive_complete_svo(self,sentence):
        # Complete-clause criterion: after stripping terminal punctuation,
        # the entire sentence must be exactly DET NOUN VERB DET NOUN.
        tokens=[x.lower() for x in TOKEN_RE.findall(sentence)]
        if len(tokens)!=5:
            return Perceived(sentence,None,None,0.0,())
        tags=[self.tag(t) for t in tokens]
        if tags!=["DET","NOUN","VERB","DET","NOUN"]:
            return Perceived(sentence,None,None,0.0,())
        if not self.learned_svo:
            return Perceived(sentence,None,None,0.0,())

        names=[]; modes=[]; states=[]
        for token in (tokens[1],tokens[2],tokens[4]):
            name,mode=self.lexicon.resolve(token)
            if name is None:
                return Perceived(sentence,None,None,0.0,())
            state=self.semantic.perceive(name,context=())
            names.append(name); modes.append(mode); states.append(state)

        frame=Frame(
            predicate=names[1],
            agent=names[0],
            patient=names[2],
        )
        confidence=min(
            s.confidence if s.committed is not None else 0.50
            for s in states
        )
        return Perceived(
            sentence,frame,"DET|NOUN|VERB",
            confidence,tuple(modes)
        )

    def generate(self,frame):
        f=frame.normalized()
        return f"the {f.agent} {f.predicate} the {f.patient}"

    def p2g2p(self,sentence):
        first=self.perceive_complete_svo(sentence)
        if first.frame is None:
            return {
                "pass":False,
                "reason":"not_a_complete_learned_svo_clause",
                "input":sentence,
                "generated":None,
                "input_frame":None,
                "roundtrip_frame":None,
            }
        generated=self.generate(first.frame)
        second=self.perceive_complete_svo(generated)
        ok=(
            second.frame is not None
            and second.frame.normalized()==first.frame.normalized()
        )
        return {
            "pass":ok,
            "reason":"ok" if ok else "reperception_failed",
            "input":sentence,
            "generated":generated,
            "input_frame":first.frame,
            "roundtrip_frame":second.frame,
            "input_confidence":first.confidence,
            "roundtrip_confidence":second.confidence,
            "lexical_modes":first.lexical_modes,
        }

    def g2p2g(self,frame):
        frame=frame.normalized()
        generated=self.generate(frame)
        perceived=self.perceive_complete_svo(generated)
        regenerated=(
            self.generate(perceived.frame)
            if perceived.frame is not None else None
        )
        return {
            "pass":(
                perceived.frame is not None
                and perceived.frame.normalized()==frame
                and regenerated==generated
            ),
            "generated":generated,
            "input_frame":frame,
            "perceived_frame":perceived.frame,
            "regenerated":regenerated,
            "confidence":perceived.confidence,
        }


def load_semantic(conceptnet):
    graph=IndexedConceptNet(conceptnet).build_index()
    memory=IndexedSemanticMemory.from_edges(
        SemanticEdge(
            source=e.source,
            relation=e.relation,
            target=e.target,
            weight=getattr(e,"weight",1.0),
            provenance="conceptnet",
        )
        for edges in graph.adj.values() for e in edges
    )
    return graph,IntegratedSemanticArchitecture(memory)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("corpus",nargs="?",type=Path,
                   default=Path(r".\data\BabyLM-2026-Strict-Small"))
    p.add_argument("--conceptnet",type=Path,
                   default=Path(r".\data\conceptnet_compact.db"))
    p.add_argument("--train-limit",type=int,default=10000)
    p.add_argument("--heldout",type=int,default=1000)
    p.add_argument("--max-cases",type=int,default=100)
    p.add_argument("--max-scan",type=int,default=10000)
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()

    if args.smoke:
        from roundtrip_benchmark import smoke
        smoke()
        print("V390 smoke wrapper: PASS")
        return

    start=time.perf_counter()
    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V390 REAL COMPLETE-CLAUSE BIDIRECTIONAL ROUNDTRIP",flush=True)
    print("="*78,flush=True)

    print("[1/8] Validating inputs...",flush=True)
    if not corpus.exists(): raise SystemExit(f"BabyLM not found: {corpus}")
    if not conceptnet.exists(): raise SystemExit(f"ConceptNet not found: {conceptnet}")

    reader=BabyLMReader()
    files=reader.files(corpus)
    print(f"      BabyLM files={len(files)}",flush=True)

    print("[2/8] Loading ConceptNet...",flush=True)
    graph,semantic=load_semantic(conceptnet)
    print(
        f"      concepts={len(graph.concepts):,} edges={graph.edge_count:,}",
        flush=True,
    )

    print("[3/8] Loading BabyLM...",flush=True)
    lines=list(reader.lines(corpus,limit=args.train_limit+args.heldout))
    train=lines[:-args.heldout] if args.heldout else lines
    heldout=lines[-args.heldout:] if args.heldout else []
    print(
        f"      train={len(train):,} heldout={len(heldout):,}",
        flush=True,
    )

    print("[4/8] Learning grammar...",flush=True)
    learner=GrammarCognitiveLearner(semantic)
    for i,sentence in enumerate(train,1):
        learner.observe_sentence(sentence,learn=True)
        if args.progress_every and (i%args.progress_every==0 or i==len(train)):
            print(
                f"      train {i:,}/{len(train):,} "
                f"rules={len(learner.grammar.memory.rules)} "
                f"observations={learner.grammar_observations} "
                f"empty={learner.empty_hypothesis_sentences}",
                flush=True,
            )

    lexicon=GraphLexicon(learner.semantic.memory)
    bench=RealRoundtrip(learner,lexicon)

    print("[5/8] Learned grammar capability...",flush=True)
    print(f"      learned_svo={bench.learned_svo}",flush=True)

    print("[6/8] Discovering complete held-out clauses...",flush=True)
    selected=[]
    scanned=0
    eligible=0
    scan_limit=min(args.max_scan,len(heldout))

    for sentence in heldout[:scan_limit]:
        scanned+=1
        r=bench.perceive_complete_svo(sentence)
        if r.frame is not None:
            eligible+=1
            selected.append((sentence,r.frame))
            if len(selected)>=args.max_cases:
                break
        if args.progress_every and scanned%args.progress_every==0:
            print(
                f"      scan {scanned:,}/{scan_limit:,} "
                f"eligible={eligible:,}",
                flush=True,
            )

    print(
        f"      scanned={scanned:,} "
        f"eligible={eligible:,}",
        flush=True,
    )

    print("[7/8] Running both roundtrip directions...",flush=True)
    p2g=[]
    g2p=[]
    for i,(sentence,frame) in enumerate(selected,1):
        p2g.append(bench.p2g2p(sentence))
        g2p.append(bench.g2p2g(frame))
        if args.progress_every and (
            i%args.progress_every==0 or i==len(selected)
        ):
            print(
                f"      cases={i:,}/{len(selected):,} "
                f"p2g={sum(int(x['pass']) for x in p2g):,}/{len(p2g):,} "
                f"g2p={sum(int(x['pass']) for x in g2p):,}/{len(g2p):,}",
                flush=True,
            )

    p2g_acc=sum(int(x["pass"]) for x in p2g)/max(1,len(p2g))
    g2p_acc=sum(int(x["pass"]) for x in g2p)/max(1,len(g2p))
    coverage=eligible/max(1,scanned)

    print("[8/8] Final checks...",flush=True)
    checks={
        "conceptnet_loaded":graph.edge_count>0,
        "babylm_loaded":bool(train and heldout),
        "grammar_learned":bool(learner.grammar.memory.rules),
        "corpus_accounting":learner.corpus_sentences_seen==len(train),
        "learned_svo_available":bench.learned_svo,
        "complete_clause_cases_found":eligible>0,
        "scan_coverage_reported":scanned>0,
        "p2g_pass":p2g_acc>=0.80,
        "g2p_pass":g2p_acc>=0.80,
    }
    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v390",
        "real_data":True,
        "conceptnet":{
            "path":str(conceptnet),
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "babylm":{
            "path":str(corpus),
            "files":len(files),
            "train_sentences":len(train),
            "heldout_sentences":len(heldout),
        },
        "grammar":{
            "rules":len(learner.grammar.memory.rules),
            "corpus_sentences_seen":learner.corpus_sentences_seen,
            "grammar_observations":learner.grammar_observations,
            "empty_hypothesis_sentences":learner.empty_hypothesis_sentences,
            "commits":learner.grammar.memory.commitments,
            "learned_svo":bench.learned_svo,
        },
        "roundtrip":{
            "requested_cases":args.max_cases,
            "max_scan":scan_limit,
            "scanned_sentences":scanned,
            "complete_clause_cases":eligible,
            "scan_coverage":coverage,
            "p2g_accuracy":p2g_acc,
            "g2p_accuracy":g2p_acc,
        },
        "checks":checks,
        "examples":{
            "p2g":p2g[:10],
            "g2p":g2p[:10],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    out=Path.cwd()/"results"/"v390_real_roundtrip.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")

    print(json.dumps(checks,indent=2),flush=True)
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)
    graph.close()


if __name__=="__main__":
    main()
