
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
VERB_FALLBACK={
    "chase","chases","eat","eats","see","sees","like","likes",
    "want","wants","make","makes","take","takes","used","use",
    "need","needs","comes","come","lose","loses","got","get",
    "put","puts","keep","keeps","find","finds","give","gives",
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
class ClauseCase:
    source_sentence:str
    clause_tokens:Tuple[str,...]
    start:int
    end:int
    frame:Frame


class GraphLexicon:
    def __init__(self,memory):
        self.memory=memory
        self.concepts=memory.concepts()

    def resolve(self,token):
        t=canonical_concept(token)
        variants=[t]
        if t.endswith("ies") and len(t)>4:
            variants.append(t[:-3]+"y")
        if t.endswith("es") and len(t)>4:
            variants.append(t[:-2])
        if t.endswith("s") and len(t)>3:
            variants.append(t[:-1])
        if t.endswith("ed") and len(t)>4:
            variants.extend((t[:-2],t[:-1]))
        if t.endswith("ing") and len(t)>5:
            variants.append(t[:-3])

        seen=set()
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                if v in self.concepts:
                    return v,(
                        "exact"
                        if v==t
                        else "morphological"
                    )
        return None,"unresolved"


class RealRoundtrip:
    """
    Learns a grammar over BabyLM, then extracts *clause-bounded* SVO cases
    from held-out conversational sentences.

    This is deliberately not a raw sliding five-token window:
      * candidate must be DET NOUN VERB DET NOUN
      * left boundary must be sentence start or a clause delimiter
      * right boundary must be sentence end or a clause delimiter

    Conjunctions are treated as boundaries because BabyLM contains long
    conversational/coordinate utterances.
    """

    LEFT_BOUNDARIES={
        ",",".",";","?","!",
        "and","but","or","so","because","while","although",
        "which","that","if","then",
    }
    RIGHT_BOUNDARIES=LEFT_BOUNDARIES

    def __init__(self,learner,lexicon):
        self.learner=learner
        self.semantic=learner.semantic
        self.lexicon=lexicon
        self.learned_svo=any(
            r.split("::",1)[0]=="DET|NOUN|VERB"
            for r in learner.grammar.memory.rules
        )

    def tokens(self,sentence):
        return [
            x.lower()
            for x in TOKEN_RE.findall(sentence)
        ]

    def tag(self,t):
        if t in DET:
            return "DET"
        if t in AUX:
            return "AUX"
        if t in VERB_FALLBACK:
            return "VERB"
        if t.endswith("ing") or t.endswith("ed"):
            return "VERB"
        return "NOUN"

    def _is_boundary(self,t):
        return t in self.LEFT_BOUNDARIES

    def extract(self,sentence):
        tokens=self.tokens(sentence)
        if not self.learned_svo:
            return []

        cases=[]
        for i in range(len(tokens)-4):
            end=i+5
            span=tokens[i:end]
            if [self.tag(t) for t in span]!=[
                "DET","NOUN","VERB","DET","NOUN"
            ]:
                continue

            left_ok=(
                i==0
                or tokens[i-1] in self.LEFT_BOUNDARIES
            )
            right_ok=(
                end==len(tokens)
                or tokens[end] in self.RIGHT_BOUNDARIES
            )

            # Do not accept immediately nested function-word sequences.
            if not (left_ok and right_ok):
                continue

            agent,_=self.lexicon.resolve(span[1])
            predicate,_=self.lexicon.resolve(span[2])
            patient,_=self.lexicon.resolve(span[4])

            if not all((agent,predicate,patient)):
                continue

            # Exercise the actual cognitive semantic architecture for the
            # extracted lexical identities.
            states=[
                self.semantic.perceive(
                    x,context=()
                )
                for x in (agent,predicate,patient)
            ]

            frame=Frame(
                predicate=predicate,
                agent=agent,
                patient=patient,
            )
            cases.append(
                ClauseCase(
                    source_sentence=sentence,
                    clause_tokens=tuple(span),
                    start=i,
                    end=end,
                    frame=frame,
                )
            )
        return cases

    def generate(self,frame):
        f=frame.normalized()
        return f"the {f.agent} {f.predicate} the {f.patient}"

    def p2g2p(self,case):
        generated=self.generate(case.frame)
        extracted=self.extract(generated)

        if len(extracted)!=1:
            return {
                "pass":False,
                "reason":"generated_clause_not_recovered_uniquely",
                "input":case.source_sentence,
                "clause":" ".join(case.clause_tokens),
                "generated":generated,
                "input_frame":case.frame,
                "roundtrip_frame":None,
            }

        recovered=extracted[0].frame
        ok=recovered.normalized()==case.frame.normalized()

        return {
            "pass":ok,
            "reason":"ok" if ok else "semantic_frame_mismatch",
            "input":case.source_sentence,
            "clause":" ".join(case.clause_tokens),
            "generated":generated,
            "input_frame":case.frame,
            "roundtrip_frame":recovered,
        }

    def g2p2g(self,frame):
        frame=frame.normalized()
        generated=self.generate(frame)
        extracted=self.extract(generated)
        recovered=extracted[0].frame if len(extracted)==1 else None
        regenerated=self.generate(recovered) if recovered else None

        return {
            "pass":(
                recovered is not None
                and recovered.normalized()==frame
                and regenerated==generated
            ),
            "generated":generated,
            "input_frame":frame,
            "perceived_frame":recovered,
            "regenerated":regenerated,
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
        for edges in graph.adj.values()
        for e in edges
    )
    return graph,IntegratedSemanticArchitecture(memory)


def smoke():
    from semantic_memory import IndexedSemanticMemory,SemanticEdge
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
    ])
    learner=GrammarCognitiveLearner(
        IntegratedSemanticArchitecture(memory)
    )
    for s in [
        "the dog chases the cat",
        "the cat chases the dog",
    ]:
        learner.observe_sentence(s,learn=True)

    bench=RealRoundtrip(
        learner,
        GraphLexicon(learner.semantic.memory),
    )
    cases=bench.extract("the dog chases the cat")
    assert len(cases)==1
    p=bench.p2g2p(cases[0])
    g=bench.g2p2g(cases[0].frame)
    assert p["pass"] and g["pass"]
    print("V391 smoke: PASS")
    print("clause-boundary extraction: PASS")
    print("learned grammar gate: PASS")
    print("semantic roundtrip: PASS")
    print("generation roundtrip: PASS")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path(r".\data\BabyLM-2026-Strict-Small"),
    )
    ap.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    ap.add_argument("--train-limit",type=int,default=10000)
    ap.add_argument("--heldout",type=int,default=1000)
    ap.add_argument("--max-cases",type=int,default=100)
    ap.add_argument("--max-scan",type=int,default=10000)
    ap.add_argument("--progress-every",type=int,default=100)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    start=time.perf_counter()
    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V391 REAL CLAUSE-BOUNDED BIDIRECTIONAL ROUNDTRIP",flush=True)
    print("="*78,flush=True)

    print("[1/8] Validating inputs...",flush=True)
    if not corpus.exists():
        raise SystemExit(f"BabyLM not found: {corpus}")
    if not conceptnet.exists():
        raise SystemExit(f"ConceptNet not found: {conceptnet}")

    reader=BabyLMReader()
    files=reader.files(corpus)
    print(f"      BabyLM files={len(files)}",flush=True)

    print("[2/8] Loading ConceptNet...",flush=True)
    graph,semantic=load_semantic(conceptnet)
    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,}",
        flush=True,
    )

    print("[3/8] Loading BabyLM...",flush=True)
    lines=list(
        reader.lines(
            corpus,
            limit=args.train_limit+args.heldout,
        )
    )
    train=lines[:-args.heldout] if args.heldout else lines
    heldout=lines[-args.heldout:] if args.heldout else []
    print(
        f"      train={len(train):,} "
        f"heldout={len(heldout):,}",
        flush=True,
    )

    print("[4/8] Learning grammar...",flush=True)
    learner=GrammarCognitiveLearner(semantic)

    for i,sentence in enumerate(train,1):
        learner.observe_sentence(sentence,learn=True)
        if args.progress_every and (
            i%args.progress_every==0
            or i==len(train)
        ):
            print(
                f"      train={i:,}/{len(train):,} "
                f"rules={len(learner.grammar.memory.rules)} "
                f"observations={learner.grammar_observations} "
                f"empty={learner.empty_hypothesis_sentences}",
                flush=True,
            )

    lexicon=GraphLexicon(learner.semantic.memory)
    bench=RealRoundtrip(learner,lexicon)

    print("[5/8] Learned grammar capability...",flush=True)
    print(
        f"      learned_svo={bench.learned_svo}",
        flush=True,
    )

    print("[6/8] Discovering clause-bounded held-out cases...",flush=True)
    scan_limit=min(args.max_scan,len(heldout))
    selected=[]
    scanned=0

    for sentence in heldout[:scan_limit]:
        scanned+=1
        cases=bench.extract(sentence)
        for case in cases:
            selected.append(case)
            if len(selected)>=args.max_cases:
                break

        if args.progress_every and scanned%args.progress_every==0:
            print(
                f"      scan={scanned:,}/{scan_limit:,} "
                f"cases={len(selected):,}",
                flush=True,
            )

        if len(selected)>=args.max_cases:
            break

    print(
        f"      scanned={scanned:,} "
        f"cases={len(selected):,}",
        flush=True,
    )

    print("[7/8] Running both roundtrip directions...",flush=True)
    p2g=[]
    g2p=[]

    for i,case in enumerate(selected,1):
        p2g.append(bench.p2g2p(case))
        g2p.append(bench.g2p2g(case.frame))

        if args.progress_every and (
            i%args.progress_every==0
            or i==len(selected)
        ):
            p=sum(int(x["pass"]) for x in p2g)
            g=sum(int(x["pass"]) for x in g2p)
            print(
                f"      cases={i:,}/{len(selected):,} "
                f"p2g={p}/{i} g2p={g}/{i}",
                flush=True,
            )

    p2g_acc=sum(int(x["pass"]) for x in p2g)/max(1,len(p2g))
    g2p_acc=sum(int(x["pass"]) for x in g2p)/max(1,len(g2p))

    print("[8/8] Final checks...",flush=True)
    checks={
        "conceptnet_loaded":graph.edge_count>0,
        "babylm_loaded":bool(train and heldout),
        "grammar_learned":bool(learner.grammar.memory.rules),
        "corpus_accounting":(
            learner.corpus_sentences_seen==len(train)
        ),
        "learned_svo_available":bench.learned_svo,
        "clause_cases_found":len(selected)>0,
        "p2g_pass":p2g_acc>=0.80,
        "g2p_pass":g2p_acc>=0.80,
    }

    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v391",
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
            "clause_cases":len(selected),
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

    out=Path.cwd()/"results"/"v391_real_roundtrip.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,indent=2,default=str),
        encoding="utf-8",
    )

    print(json.dumps(checks,indent=2),flush=True)
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)

    graph.close()


if __name__=="__main__":
    main()
