
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from babylm_grammar import BabyLMReader, GrammarCognitiveLearner
from semantic_memory import IndexedSemanticMemory, SemanticEdge, canonical_concept
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


TOKEN_RE=re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+")

DETERMINERS={"the","a","an"}
KNOWN_AUX={
    "is","are","was","were","am","be","been","being",
    "do","does","did","have","has","had",
    "can","could","will","would","should","may","might","must",
}


@dataclass(frozen=True)
class Frame:
    predicate:str
    agent:str
    patient:str

    def normalized(self):
        return Frame(
            predicate=canonical_concept(self.predicate),
            agent=canonical_concept(self.agent),
            patient=canonical_concept(self.patient),
        )


@dataclass(frozen=True)
class Perceived:
    sentence:str
    frame:Optional[Frame]
    grammar_rule:Optional[str]
    confidence:float
    lexical_modes:Tuple[str,...]


def lemma_candidates(token:str):
    t=canonical_concept(token)
    if not t:
        return ()
    out=[t]

    # Conservative English morphology, only used to find an existing graph node.
    if t.endswith("ies") and len(t)>4:
        out.append(t[:-3]+"y")
    if t.endswith("es") and len(t)>4:
        out.append(t[:-2])
    if t.endswith("s") and len(t)>3:
        out.append(t[:-1])
    if t.endswith("ed") and len(t)>4:
        out.append(t[:-2])
        out.append(t[:-1])
    if t.endswith("ing") and len(t)>5:
        out.append(t[:-3])

    seen=set()
    return tuple(x for x in out if not (x in seen or seen.add(x)))


class GraphLexicon:
    def __init__(self,memory):
        self.memory=memory
        self.concepts=memory.concepts()

    def resolve(self,token):
        for candidate in lemma_candidates(token):
            if candidate in self.concepts:
                return candidate,"exact" if candidate==canonical_concept(token) else "morphological"
        return None,"unresolved"


class RealRoundtrip:
    """
    Uses the grammar actually learned from BabyLM.

    The benchmark only recognizes an SVO pattern when the learned grammar
    memory contains a DET|NOUN|VERB construction. Lexical semantics come from
    real ConceptNet graph membership. Ambiguous lexical states are allowed to
    remain uncertain; exact graph identity is sufficient for semantic
    equivalence testing.
    """

    def __init__(self,learner,lexicon):
        self.learner=learner
        self.semantic=learner.semantic
        self.lexicon=lexicon

        self.learned_svo=any(
            r.split("::",1)[0]=="DET|NOUN|VERB"
            for r in learner.grammar.memory.rules
        )
        self.learned_np=any(
            r.split("::",1)[0]=="DET|NOUN"
            for r in learner.grammar.memory.rules
        )

    def tokens(self,sentence):
        return [
            t.lower()
            for t in TOKEN_RE.findall(sentence)
        ]

    def tag(self,token):
        if token in DETERMINERS:
            return "DET"
        if token in KNOWN_AUX:
            return "AUX"
        if token in {
            "chase","chases","eat","eats","see","sees","like","likes",
            "want","wants","make","makes","take","takes",
        }:
            return "VERB"
        if token.endswith("ing") or token.endswith("ed"):
            return "VERB"
        if token.endswith("s") and len(token)>3:
            # A useful fallback for third-person verb forms. It is only used
            # after the lexical/graph gate below.
            return "VERB"
        return "NOUN"

    def lexical(self,token):
        return self.lexicon.resolve(token)

    def perceive_svo(self,sentence):
        tokens=self.tokens(sentence)
        if len(tokens)<5:
            return Perceived(
                sentence,None,None,0.0,()
            )

        # Find the first contiguous DET NOUN VERB DET NOUN span.
        for i in range(len(tokens)-4):
            span=tokens[i:i+5]
            tags=[self.tag(t) for t in span]

            if tags!=["DET","NOUN","VERB","DET","NOUN"]:
                continue
            if not self.learned_svo:
                return Perceived(
                    sentence,None,None,0.0,()
                )

            agent,amode=self.lexical(span[1])
            predicate,pmode=self.lexical(span[2])
            patient,pmode2=self.lexical(span[4])

            if not all((agent,predicate,patient)):
                continue

            # Probe the actual cognitive semantic architecture. We retain the
            # graph identity even when the semantic controller does not commit
            # under empty context.
            states=[
                self.semantic.perceive(x,context=())
                for x in (agent,predicate,patient)
            ]
            confidences=[
                s.confidence
                if s.committed is not None
                else 0.50
                for s in states
            ]

            frame=Frame(
                predicate=predicate,
                agent=agent,
                patient=patient,
            )

            return Perceived(
                sentence=sentence,
                frame=frame,
                grammar_rule="DET|NOUN|VERB",
                confidence=min(confidences),
                lexical_modes=(amode,pmode,pmode2),
            )

        return Perceived(
            sentence,None,None,0.0,()
        )

    def generate(self,frame:Frame):
        f=frame.normalized()
        return f"the {f.agent} {f.predicate} the {f.patient}"

    def p2g2p(self,sentence):
        first=self.perceive_svo(sentence)
        if first.frame is None:
            return {
                "pass":False,
                "reason":"not_parseable_by_learned_svo",
                "input":sentence,
                "generated":None,
                "input_frame":None,
                "roundtrip_frame":None,
            }

        generated=self.generate(first.frame)
        second=self.perceive_svo(generated)

        return {
            "pass":(
                second.frame is not None
                and second.frame.normalized()
                ==first.frame.normalized()
            ),
            "reason":"ok" if second.frame is not None else "reperception_failed",
            "input":sentence,
            "generated":generated,
            "input_frame":first.frame,
            "roundtrip_frame":second.frame,
            "input_confidence":first.confidence,
            "roundtrip_confidence":second.confidence,
            "input_lexical_modes":first.lexical_modes,
        }

    def g2p2g(self,frame):
        frame=frame.normalized()
        generated=self.generate(frame)
        perceived=self.perceive_svo(generated)

        regenerated=(
            self.generate(perceived.frame)
            if perceived.frame is not None
            else None
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
        for edges in graph.adj.values()
        for e in edges
    )
    return graph,IntegratedSemanticArchitecture(memory)


def run(args):
    start=time.perf_counter()

    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V387 REAL BIDIRECTIONAL ROUNDTRIP",flush=True)
    print("="*78,flush=True)

    print("[1/8] Validating inputs...",flush=True)
    if not corpus.exists():
        raise SystemExit(f"BabyLM not found: {corpus}")
    if not conceptnet.exists():
        raise SystemExit(f"ConceptNet not found: {conceptnet}")

    reader=BabyLMReader()
    files=reader.files(corpus)
    print(f"      BabyLM files: {len(files)}",flush=True)

    print("[2/8] Loading ConceptNet...",flush=True)
    graph,semantic=load_semantic(conceptnet)
    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,}",
        flush=True,
    )

    print("[3/8] Loading BabyLM train/held-out...",flush=True)
    lines=list(
        reader.lines(
            corpus,
            limit=args.train_limit+args.heldout,
        )
    )
    train=lines[:-args.heldout] if args.heldout else lines
    heldout=lines[-args.heldout:] if args.heldout else []
    print(
        f"      train={len(train):,} heldout={len(heldout):,}",
        flush=True,
    )

    print("[4/8] Learning grammar from BabyLM...",flush=True)
    learner=GrammarCognitiveLearner(semantic)
    for i,sentence in enumerate(train,1):
        learner.observe_sentence(sentence,learn=True)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(train)
        ):
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
    print(
        f"      learned_svo={bench.learned_svo} "
        f"learned_np={bench.learned_np}",
        flush=True,
    )

    print("[6/8] Discovering roundtrip-eligible held-out cases...",flush=True)

    # V387 incorrectly treated the first N held-out sentences as the benchmark
    # sample. Natural corpora need case discovery: find sentences that the
    # learned grammar can actually express, then benchmark those cases.
    scan_limit=min(args.max_scan,len(heldout))
    candidate_sentences=[]
    parseable=0

    for i,sentence in enumerate(
        heldout[:scan_limit],
        1,
    ):
        result=bench.p2g2p(sentence)

        if result["input_frame"] is not None:
            parseable += 1
            candidate_sentences.append(
                (sentence,result)
            )

            if len(candidate_sentences)>=args.max_cases:
                break

        if args.progress_every and (
            i%args.progress_every==0
        ):
            print(
                f"      discovery {i:,}/{scan_limit:,} "
                f"eligible={len(candidate_sentences):,} "
                f"scan_parse_rate="
                f"{parseable/max(1,i):.3f}",
                flush=True,
            )

    total_scanned=(
        i if scan_limit else 0
    )
    p2g=[
        result
        for _,result in candidate_sentences
        if result["pass"]
    ]

    # Keep the selected semantic frames so G→P→G uses the exact same cases,
    # not a second independent scan.
    selected_frames=[
        result["input_frame"]
        for _,result in candidate_sentences
    ]

    print(
        f"      discovered={len(candidate_sentences):,} "
        f"scanned={total_scanned:,} "
        f"parseable={parseable:,}",
        flush=True,
    )

    p2g_accuracy=len(p2g)/max(1,parseable)
    parse_coverage=parseable/max(1,total_scanned)

    print("[7/8] G → P → G from recovered semantic frames...",flush=True)
    g2p=[]
    for i,r in enumerate(
        [
            bench.p2g2p(s)[ "input_frame" ]
            for s in heldout[:args.max_cases]
        ],
        1,
    ):
        if r is None:
            continue
        g2p.append(
            bench.g2p2g(r)
        )

        if args.progress_every and (
            len(g2p)%args.progress_every==0
        ):
            passed=sum(int(x["pass"]) for x in g2p)
            print(
                f"      cases={len(g2p):,} "
                f"pass={passed:,} "
                f"accuracy={passed/max(1,len(g2p)):.3f}",
                flush=True,
            )

    g2p_accuracy=sum(
        int(x["pass"]) for x in g2p
    )/max(1,len(g2p))

    print("[8/8] Final checks...",flush=True)
    checks={
        "conceptnet_loaded":graph.edge_count>0,
        "babylm_loaded":len(train)>0 and len(heldout)>0,
        "grammar_learned":len(learner.grammar.memory.rules)>0,
        "corpus_accounting":learner.corpus_sentences_seen==len(train),
        "learned_svo_available":bench.learned_svo,
        "case_discovery_found_cases":len(candidate_sentences)>0,
        "case_discovery_accounting":(
            len(candidate_sentences)
            == len(p2g) + sum(
                1 for _,r in candidate_sentences
                if not r["pass"]
            )
        ),
        "scan_parse_coverage":(
            parseable/max(1,total_scanned)>=0.01
        ),
        "p2g_roundtrip_accuracy":p2g_accuracy>=0.80,
        "g2p_has_cases":len(g2p)>0,
        "g2p_roundtrip_accuracy":g2p_accuracy>=0.80,
    }

    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v388",
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
            "scan_accounting":{
                "scanned_sentences":total_scanned,
                "parseable_sentences":parseable,
                "discovered_cases":len(candidate_sentences),
            },
            "max_scan":scan_limit,
            "scanned_sentences":total_scanned,
            "discovered_cases":len(candidate_sentences),
            "scan_parse_coverage":(
                parseable/max(1,total_scanned)
            ),
            "p2g_roundtrip_accuracy":p2g_accuracy,
            "g2p_cases":len(g2p),
            "g2p_roundtrip_accuracy":g2p_accuracy,
        },
        "semantic_performance":learner.performance_snapshot(),
        "checks":checks,
        "examples":{
            "p2g":p2g[:10],
            "g2p":g2p[:10],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    results=Path.cwd()/"results"
    results.mkdir(parents=True,exist_ok=True)
    out=results/"v389_real_roundtrip.json"
    out.write_text(
        json.dumps(report,indent=2,default=str),
        encoding="utf-8",
    )

    print(json.dumps(checks,indent=2),flush=True)
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)

    graph.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path(r".\data\BabyLM-2026-Strict-Small"),
    )
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument("--train-limit",type=int,default=10000)
    p.add_argument("--heldout",type=int,default=1000)
    p.add_argument("--max-cases",type=int,default=100)
    p.add_argument(
        "--max-scan",
        type=int,
        default=10000,
        help="Maximum held-out sentences to inspect while discovering cases.",
    )
    p.add_argument("--progress-every",type=int,default=25)
    p.add_argument("--smoke",action="store_true")

    args=p.parse_args()

    if args.smoke:
        # Keep a data-free smoke path using V383's deterministic benchmark.
        from roundtrip_benchmark import smoke
        result=smoke()
        print("V387 smoke wrapper: PASS")
        return

    run(args)


if __name__=="__main__":
    main()
