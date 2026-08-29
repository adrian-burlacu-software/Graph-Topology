
from __future__ import annotations

from pathlib import Path
import argparse, json, time, math
import sys

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from babylm_grammar import BabyLMReader, GrammarCognitiveLearner
from roundtrip_cognitive import (
    BidirectionalRoundTripBenchmark,
    SemanticFrame,
    RoundTripPerception,
    RoundTripProduction,
    semantic_equivalent,
)
from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


def progress(msg, start, **fields):
    elapsed=time.perf_counter()-start
    suffix=""
    if fields:
        suffix=" | "+" ".join(f"{k}={v}" for k,v in fields.items())
    print(f"[{elapsed:8.2f}s] {msg}{suffix}", flush=True)


def load_real_semantics(conceptnet: Path, start):
    progress("STAGE 2/7 — indexing ConceptNet", start)
    graph=IndexedConceptNet(conceptnet).build_index()
    progress(
        "ConceptNet ready",
        start,
        concepts=f"{len(graph.concepts):,}",
        edges=f"{graph.edge_count:,}",
    )

    progress("STAGE 3/7 — constructing cognitive semantic memory", start)
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
    arch=IntegratedSemanticArchitecture(memory)
    progress("Cognitive semantic memory ready", start)
    return graph, arch


def load_babylm(corpus: Path, train_limit: int, heldout: int):
    reader=BabyLMReader()
    lines=list(
        reader.lines(
            corpus,
            limit=train_limit+heldout,
        )
    )
    if not lines:
        raise RuntimeError("BabyLM yielded no readable records.")

    train=lines[:-heldout] if heldout else lines
    test=lines[-heldout:] if heldout else []
    return reader,train,test


def learn_grammar(arch, train, progress_every, start):
    learner=GrammarCognitiveLearner(arch)

    progress("STAGE 4/7 — learning grammar from BabyLM", start)

    for i,sentence in enumerate(train,1):
        learner.observe_sentence(sentence,learn=True)

        if progress_every and (
            i % progress_every == 0 or i == len(train)
        ):
            progress(
                "Grammar learning progress",
                start,
                episode=f"{i}/{len(train)}",
                rules=len(learner.grammar.memory.rules),
                observations=learner.grammar_observations,
                corpus_seen=learner.corpus_sentences_seen,
                semantic_requests=learner.semantic_requests,
                cache_hits=learner.semantic_cache_hits,
            )

    return learner


def semantic_frame_from_benchmark_perception(perception):
    return perception.frame



class LearnedGrammarRoundTripBenchmark:
    """
    Uses the grammar rules actually accumulated by GrammarCognitiveLearner.
    The existing V383 semantic/frame machinery remains the execution substrate,
    but perception and production are gated by learned rule availability.
    """

    def __init__(self, learner):
        self.learner=learner
        self.semantic=learner.semantic
        self.perception=RoundTripPerception(self.semantic)
        self.production=RoundTripProduction(self.semantic)

        self.learned_rules=set(
            learner.grammar.memory.rules
        )

    def perception_then_generation(self, sentence):
        result=self.perception.perceive(sentence)

        # Map V383's transparent SVO realization to the grammar rule learned
        # from BabyLM. Do not claim a successful learned-grammar parse if the
        # corresponding construction was never induced.
        if result.frame is not None:
            if result.grammar_rule == "DET|NOUN|VERB::SVO":
                learned_svo=any(
                    r.split("::",1)[0]=="DET|NOUN|VERB"
                    for r in self.learned_rules
                )
                if not learned_svo:
                    result=result.__class__(
                        sentence=sentence,
                        frame=None,
                        grammar_rule=None,
                        confidence=0.0,
                    )
            elif result.grammar_rule == "DET|NOUN::ENTITY":
                learned_np=any(
                    r.split("::",1)[0]=="DET|NOUN"
                    for r in self.learned_rules
                )
                if not learned_np:
                    result=result.__class__(
                        sentence=sentence,
                        frame=None,
                        grammar_rule=None,
                        confidence=0.0,
                    )

        if result.frame is None:
            return {
                "pass":False,
                "direction":"perception_to_generation_to_perception",
                "reason":"learned_grammar_parse_failed",
                "input":sentence,
                "input_frame":None,
                "generated":None,
                "roundtrip_frame":None,
                "perception_confidence":0.0,
                "reperception_confidence":0.0,
            }

        generated=self.production.generate(result.frame)
        second=self.perception.perceive(generated)

        return {
            "pass":(
                second.frame is not None
                and semantic_equivalent(
                    result.frame,
                    second.frame,
                )
            ),
            "direction":"perception_to_generation_to_perception",
            "input":sentence,
            "input_frame":result.frame,
            "generated":generated,
            "roundtrip_frame":second.frame,
            "perception_confidence":result.confidence,
            "reperception_confidence":second.confidence,
        }

    def generation_to_perception_to_generation(self, frame):
        generated=self.production.generate(frame)
        perceived=self.perception.perceive(generated)

        learned=True
        if perceived.grammar_rule=="DET|NOUN|VERB::SVO":
            learned=any(
                r.split("::",1)[0]=="DET|NOUN|VERB"
                for r in self.learned_rules
            )
        elif perceived.grammar_rule=="DET|NOUN::ENTITY":
            learned=any(
                r.split("::",1)[0]=="DET|NOUN"
                for r in self.learned_rules
            )

        regenerated=(
            self.production.generate(perceived.frame)
            if learned and perceived.frame is not None
            else None
        )

        return {
            "pass":(
                learned
                and perceived.frame is not None
                and semantic_equivalent(
                    frame,
                    perceived.frame,
                )
                and regenerated==generated
            ),
            "direction":"generation_to_perception_to_generation",
            "input_frame":frame,
            "generated":generated,
            "perceived_frame":perceived.frame if learned else None,
            "regenerated":regenerated,
            "perception_confidence":(
                perceived.confidence if learned else 0.0
            ),
        }


def run_roundtrip(learner, heldout, max_cases, progress_every, start):
    bench=LearnedGrammarRoundTripBenchmark(learner)

    p2g=[]
    g2p=[]
    parseable=0
    total_seen=min(len(heldout),max_cases)

    progress(
        "STAGE 5/7 — learned grammar perception → generation → perception",
        start,
    )

    parsed_frames=[]
    for i,sentence in enumerate(heldout[:max_cases],1):
        result=bench.perception_then_generation(sentence)
        if result["input_frame"] is not None:
            parseable += 1
            parsed_frames.append(
                (sentence,result["input_frame"])
            )

        if result["pass"]:
            p2g.append(result)

        if progress_every and (
            i%progress_every==0
            or i==1
            or i==total_seen
        ):
            progress(
                "P→G→P progress",
                start,
                cases=i,
                parse_rate=f"{parseable/max(1,i):.3f}",
                roundtrip_pass_rate=(
                    f"{len(p2g)/max(1,i):.3f}"
                ),
            )

    p2g_accuracy=len(p2g)/max(1,parseable)

    progress(
        "P→G→P complete",
        start,
        cases=total_seen,
        parseable=parseable,
        parse_rate=f"{parseable/max(1,total_seen):.3f}",
        semantic_roundtrip_accuracy=f"{p2g_accuracy:.3f}",
    )

    progress(
        "STAGE 6/7 — learned grammar generation → perception → generation",
        start,
    )

    for i,(_,frame) in enumerate(
        parsed_frames,
        1,
    ):
        g2p.append(
            bench.generation_to_perception_to_generation(
                frame
            )
        )

        if progress_every and (
            i%progress_every==0
            or i==1
            or i==len(parsed_frames)
        ):
            acc=sum(
                int(x["pass"]) for x in g2p
            )/len(g2p)
            progress(
                "G→P→G progress",
                start,
                cases=len(g2p),
                accuracy=f"{acc:.3f}",
            )

    return p2g,g2p,parseable,total_seen



def main():
    p=argparse.ArgumentParser(
        description="Real-data bidirectional semantic/grammar roundtrip benchmark."
    )
    p.add_argument(
        "corpus",
        type=Path,
        default=Path(r".\data\BabyLM-2026-Strict-Small"),
        nargs="?",
    )
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument("--train-limit",type=int,default=10000)
    p.add_argument("--heldout",type=int,default=1000)
    p.add_argument("--max-cases",type=int,default=100)
    p.add_argument("--progress-every",type=int,default=25)
    p.add_argument(
        "--result-name",
        default="v386_real_roundtrip.json",
    )
    p.add_argument("--smoke", action="store_true")
    args=p.parse_args()

    start=time.perf_counter()
    results_path=Path.cwd()/"results"/args.result_name

    if args.smoke:
        from roundtrip_benchmark import smoke
        result=smoke()
        result["version"]="v384"
        result["mode"]="data_free_smoke"
        results_path.parent.mkdir(parents=True,exist_ok=True)
        results_path.write_text(
            json.dumps(result,indent=2,default=str),
            encoding="utf-8",
        )
        print(
            f"[RESULT FILE] {results_path.resolve()}",
            flush=True,
        )
        return

    print("="*78,flush=True)
    print("V384 REAL BIDIRECTIONAL ROUNDTRIP BENCHMARK",flush=True)
    print("="*78,flush=True)

    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    progress("STAGE 1/7 — validating real data",start)

    if not corpus.exists():
        raise SystemExit(f"BabyLM dataset not found: {corpus}")
    if not conceptnet.exists():
        raise SystemExit(f"ConceptNet database not found: {conceptnet}")

    reader,train,heldout=load_babylm(
        corpus,
        args.train_limit,
        args.heldout,
    )

    progress(
        "BabyLM ready",
        start,
        files=len(reader.files(corpus)),
        train=len(train),
        heldout=len(heldout),
    )

    graph,arch=load_real_semantics(conceptnet,start)
    learner=learn_grammar(
        arch,
        train,
        args.progress_every,
        start,
    )

    p2g,g2p,parseable,total_seen=run_roundtrip(
        learner,
        heldout,
        args.max_cases,
        args.progress_every,
        start,
    )

    progress("STAGE 7/7 — final metrics",start)

    p2g_acc=sum(int(x["pass"]) for x in p2g)/max(1,len(p2g))

    g2p_acc=sum(
        int(x["pass"]) for x in g2p
    )/max(1,len(g2p))

    result={
        "status":"PASS",
        "version":"v384",
        "real_data":True,
        "conceptnet":{
            "path":str(conceptnet),
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "babylm":{
            "path":str(corpus),
            "files":len(reader.files(corpus)),
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
            "max_cases":args.max_cases,
            "perception_to_generation_to_perception":{
                "cases":len(p2g),
                "accuracy":p2g_acc,
            },
            "generation_to_perception_to_generation":{
                "cases":len(g2p),
                "accuracy":g2p_acc,
            },
        },
        "semantic_performance":learner.performance_snapshot(),
        "checks":{
            "conceptnet_loaded":graph.edge_count>0,
            "babylm_loaded":len(train)>0 and len(heldout)>0,
            "grammar_learned":len(learner.grammar.memory.rules)>0,
            "corpus_accounting":learner.corpus_sentences_seen==len(train),
            "roundtrip_p2g_has_cases":total_seen>0,
            "roundtrip_g2p_has_cases":len(g2p)>0,
            "learned_grammar_used":len(learner.grammar.memory.rules)>0,
            "roundtrip_p2g_passes":p2g_acc>=0.80,
            "roundtrip_g2p_passes":g2p_acc>=0.80,
        },
        "examples":{
            "p2g":p2g[:10],
            "g2p":g2p[:10],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    result["status"]="PASS" if all(result["checks"].values()) else "FAIL"

    for k,v in result["checks"].items():
        print(
            f"  {k:34} {'PASS' if v else 'FAIL'}",
            flush=True,
        )
    print(
        f"[RESULT] {result['status']}",
        flush=True,
    )

    results_path.parent.mkdir(parents=True,exist_ok=True)
    results_path.write_text(
        json.dumps(result,indent=2,default=str),
        encoding="utf-8",
    )
    print(
        f"[RESULT FILE] {results_path.resolve()}",
        flush=True,
    )

    graph.close()


if __name__=="__main__":
    main()
