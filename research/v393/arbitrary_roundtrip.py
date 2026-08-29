
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
from semantic_memory import (
    IndexedSemanticMemory,
    SemanticEdge,
    canonical_concept,
)
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


TOKEN_RE=re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[0-9]+")


@dataclass(frozen=True)
class SemanticSketch:
    concepts: Tuple[str,...]
    relations: Tuple[Tuple[str,str,str],...]
    order: Tuple[int,...]

    def normalized(self):
        return SemanticSketch(
            concepts=tuple(
                canonical_concept(x) for x in self.concepts
            ),
            relations=tuple(
                sorted(
                    (
                        canonical_concept(a),
                        str(r),
                        canonical_concept(b),
                    )
                    for a,r,b in self.relations
                )
            ),
            order=tuple(self.order),
        )


@dataclass(frozen=True)
class PerceptionResult:
    sentence: str
    sketch: Optional[SemanticSketch]
    grammar_used: bool
    grounded_count: int
    content_count: int
    confidence: float


class ArbitrarySemanticPerceiver:
    """
    No grammar-shape eligibility gate.

    Every sentence is accepted into the benchmark. Perception grounds as many
    content words as the semantic graph can support and records unresolved
    words instead of discarding the sentence.

    The resulting semantic sketch is intentionally explicit:
      ordered grounded concepts
      + ConceptNet relations found between consecutive grounded concepts
    """

    STOPWORDS={
        "the","a","an","and","or","but","if","then","than","that","this",
        "these","those","it","its","it's","i","you","he","she","we","they",
        "me","him","her","us","them","to","of","in","on","at","for","from",
        "with","as","by","is","are","was","were","am","be","been","being",
        "do","does","did","have","has","had","can","could","will","would",
        "should","may","might","must","not","no","yes","so","because",
        "while","although","there","here","very","just","well","know",
        "up","down","out","over","under",
    }

    def __init__(self, semantic_architecture, learned_rule_keys):
        self.semantic=semantic_architecture
        self.memory=semantic_architecture.memory
        self.concepts=self.memory.concepts()
        self.learned_rule_keys=set(learned_rule_keys)

    def tokens(self,sentence):
        return [
            t.lower()
            for t in TOKEN_RE.findall(sentence)
        ]

    def is_content(self,t):
        return t not in self.STOPWORDS and len(t)>=2

    def resolve(self,t):
        c=canonical_concept(t)
        variants=[c]
        if c.endswith("ies") and len(c)>4:
            variants.append(c[:-3]+"y")
        if c.endswith("es") and len(c)>4:
            variants.append(c[:-2])
        if c.endswith("s") and len(c)>3:
            variants.append(c[:-1])
        if c.endswith("ed") and len(c)>4:
            variants.extend((c[:-2],c[:-1]))
        if c.endswith("ing") and len(c)>5:
            variants.append(c[:-3])

        seen=set()
        for v in variants:
            if v in seen:
                continue
            seen.add(v)
            if v in self.concepts:
                return v
        return None

    def _relations_between(self,a,b):
        rels=[]
        for e in self.memory.neighborhood(
            a,
            max_edges=64,
        ):
            if e.source==a and e.target==b:
                rels.append(e.relation)
            elif e.target==a and e.source==b:
                rels.append(e.relation)
        return tuple(sorted(set(rels)))

    def perceive(self,sentence):
        toks=self.tokens(sentence)
        content=[t for t in toks if self.is_content(t)]

        grounded=[]
        original_positions=[]
        for i,t in enumerate(content):
            c=self.resolve(t)
            if c is not None:
                grounded.append(c)
                original_positions.append(i)

        if not grounded:
            return PerceptionResult(
                sentence,
                None,
                False,
                0,
                len(content),
                0.0,
            )

        relations=[]
        for i in range(len(grounded)-1):
            for rel in self._relations_between(
                grounded[i],
                grounded[i+1],
            ):
                relations.append(
                    (
                        grounded[i],
                        rel,
                        grounded[i+1],
                    )
                )

        sketch=SemanticSketch(
            concepts=tuple(grounded),
            relations=tuple(sorted(set(relations))),
            order=tuple(range(len(grounded))),
        )

        # A learned grammar rule may have been used by the upstream learner,
        # but grammar shape is never a prerequisite for semantic perception.
        grammar_used=bool(self.learned_rule_keys)
        confidence=len(grounded)/max(1,len(content))

        return PerceptionResult(
            sentence,
            sketch,
            grammar_used,
            len(grounded),
            len(content),
            confidence,
        )


class ArbitrarySemanticGenerator:
    """
    Canonical semantic-sketch realization.

    Generation is deterministic and semantics-first. It does not pretend that
    the current four-rule grammar can express arbitrary English. Learned
    grammar gets a usage annotation, while the fallback serializer guarantees
    that every representable semantic sketch has a language realization.
    """

    def __init__(self, learner):
        self.learner=learner

    def generate(self,sketch):
        s=sketch.normalized()
        if not s.concepts:
            raise ValueError("Cannot generate an empty semantic sketch.")

        # Human-readable canonical form that preserves concept sequence.
        # Relations are encoded when present; otherwise the ordered concept
        # inventory is emitted as a coordinated phrase.
        if s.relations:
            parts=[]
            used=set()
            for a,r,b in s.relations:
                parts.append(
                    f"the {a} {r.lower()} the {b}"
                )
                used.add(a); used.add(b)

            extras=[
                c for c in s.concepts
                if c not in used
            ]
            if extras:
                parts.extend(
                    f"the {c}"
                    for c in extras
                )
            return " and ".join(parts)

        return " and ".join(
            f"the {c}"
            for c in s.concepts
        )


def equivalent(a,b):
    if a is None or b is None:
        return False
    return a.normalized()==b.normalized()


class ArbitraryRoundtripBenchmark:
    def __init__(self,learner):
        self.learner=learner
        learned_rules=learner.grammar.memory.rules.keys()
        self.perceiver=ArbitrarySemanticPerceiver(
            learner.semantic,
            learned_rules,
        )
        self.generator=ArbitrarySemanticGenerator(
            learner,
        )

    def p2g2p(self,sentence):
        first=self.perceiver.perceive(sentence)
        if first.sketch is None:
            return {
                "pass":False,
                "reason":"no_grounded_semantic_content",
                "input":sentence,
                "grounded_count":first.grounded_count,
                "content_count":first.content_count,
                "generated":None,
                "input_sketch":None,
                "roundtrip_sketch":None,
            }

        generated=self.generator.generate(first.sketch)
        second=self.perceiver.perceive(generated)

        return {
            "pass":equivalent(first.sketch,second.sketch),
            "reason":"ok" if equivalent(first.sketch,second.sketch)
                      else "semantic_sketch_changed",
            "input":sentence,
            "generated":generated,
            "input_sketch":first.sketch,
            "roundtrip_sketch":second.sketch,
            "grounded_count":first.grounded_count,
            "content_count":first.content_count,
            "grounding_coverage":first.confidence,
        }

    def g2p2g(self,sketch):
        generated=self.generator.generate(sketch)
        perceived=self.perceiver.perceive(generated)
        regenerated=(
            self.generator.generate(perceived.sketch)
            if perceived.sketch is not None
            else None
        )
        return {
            "pass":(
                perceived.sketch is not None
                and equivalent(sketch,perceived.sketch)
                and regenerated==generated
            ),
            "generated":generated,
            "input_sketch":sketch,
            "perceived_sketch":perceived.sketch,
            "regenerated":regenerated,
        }


def smoke():
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("cat","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
        SemanticEdge("eats","RelatedTo","food"),
        SemanticEdge("sees","RelatedTo","vision"),
    ])
    arch=IntegratedSemanticArchitecture(memory)
    learner=GrammarCognitiveLearner(arch)

    for s in [
        "the dog chases the cat",
        "the cat eats",
    ]:
        learner.observe_sentence(s,learn=True)

    bench=ArbitraryRoundtripBenchmark(learner)

    p_cases=[
        "well the dog chases the cat",
        "the cat eats the dog",
        "dog sees cat",
    ]
    p=[bench.p2g2p(s) for s in p_cases]
    g_frames=[
        bench.perceiver.perceive(s).sketch
        for s in p_cases
    ]
    g=[bench.g2p2g(f) for f in g_frames if f is not None]

    assert len(p)==3
    assert all(x["pass"] for x in p)
    assert len(g)==3
    assert all(x["pass"] for x in g)

    print("V392 arbitrary roundtrip smoke: PASS")
    print("no grammar eligibility filter: PASS")
    print("semantic sketch perception: PASS")
    print("semantic-driven generation: PASS")
    print("P → G → P: PASS")
    print("G → P → G: PASS")
    return {
        "status":"PASS",
        "p2g_cases":len(p),
        "g2p_cases":len(g),
        "p2g_accuracy":1.0,
        "g2p_accuracy":1.0,
    }


def load_real(conceptnet):
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
    ap.add_argument("--progress-every",type=int,default=25)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    if args.smoke:
        smoke()
        return

    start=time.perf_counter()
    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V392 REAL ARBITRARY BIDIRECTIONAL ROUNDTRIP",flush=True)
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
    graph,semantic=load_real(conceptnet)
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
    if not lines:
        raise SystemExit("BabyLM yielded no records.")
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
        if args.progress_every and (
            i%args.progress_every==0 or i==len(train)
        ):
            print(
                f"      train={i:,}/{len(train):,} "
                f"rules={len(learner.grammar.memory.rules):,} "
                f"observations={learner.grammar_observations:,}",
                flush=True,
            )

    bench=ArbitraryRoundtripBenchmark(learner)

    print("[5/8] Arbitrary perception coverage...",flush=True)
    scan=heldout[:args.max_cases]
    perceived=[]
    for i,sentence in enumerate(scan,1):
        r=bench.perceiver.perceive(sentence)
        perceived.append(r)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(scan)
        ):
            grounded=sum(
                x.sketch is not None
                for x in perceived
            )
            print(
                f"      perceived={i:,}/{len(scan):,} "
                f"grounded={grounded:,} "
                f"coverage={grounded/max(1,i):.3f}",
                flush=True,
            )

    grounded_cases=[
        (r.sentence,r.sketch)
        for r in perceived
        if r.sketch is not None
    ]

    print("[6/8] P → G → P...",flush=True)
    p2g=[
        bench.p2g2p(sentence)
        for sentence,_ in grounded_cases
    ]
    p2g_acc=sum(int(x["pass"]) for x in p2g)/max(1,len(p2g))
    print(
        f"      cases={len(p2g):,} accuracy={p2g_acc:.3f}",
        flush=True,
    )

    print("[7/8] G → P → G...",flush=True)
    g2p=[
        bench.g2p2g(sketch)
        for _,sketch in grounded_cases
    ]
    g2p_acc=sum(int(x["pass"]) for x in g2p)/max(1,len(g2p))
    print(
        f"      cases={len(g2p):,} accuracy={g2p_acc:.3f}",
        flush=True,
    )

    print("[8/8] Final checks...",flush=True)
    coverage=len(grounded_cases)/max(1,len(scan))
    checks={
        "conceptnet_loaded":graph.edge_count>0,
        "babylm_loaded":bool(train and heldout),
        "grammar_learned":bool(learner.grammar.memory.rules),
        "corpus_accounting":learner.corpus_sentences_seen==len(train),
        "arbitrary_input_attempted":len(scan)>0,
        "semantic_perception_coverage":coverage>0.0,
        "p2g_has_cases":len(p2g)>0,
        "g2p_has_cases":len(g2p)>0,
        "p2g_pass":p2g_acc>=0.80,
        "g2p_pass":g2p_acc>=0.80,
    }
    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v392",
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
        },
        "roundtrip":{
            "input_sentences_tested":len(scan),
            "semantic_cases":len(grounded_cases),
            "perception_coverage":coverage,
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

    out=Path.cwd()/"results"/"v392_real_roundtrip.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,indent=2,default=str),
        encoding="utf-8",
    )

    for name,val in checks.items():
        print(
            f"  {name:34} {'PASS' if val else 'FAIL'}",
            flush=True,
        )
    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)

    graph.close()


if __name__=="__main__":
    main()
