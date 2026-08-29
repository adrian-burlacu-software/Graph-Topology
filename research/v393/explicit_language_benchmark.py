
from __future__ import annotations

import argparse, json, time, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

from explicit_language_model import ExplicitRoundtrip
from babylm_grammar import BabyLMReader, GrammarCognitiveLearner
from semantic_memory import IndexedSemanticMemory, SemanticEdge
from semantic_architecture import IntegratedSemanticArchitecture
from real_grounding import IndexedConceptNet


def smoke():
    from explicit_language_model import smoke as run
    result=run()
    print(json.dumps(result,indent=2))
    return result


def real_run(args):
    start=time.perf_counter()
    corpus=args.corpus.resolve()
    db=args.conceptnet.resolve()

    print("="*78,flush=True)
    print("V393 EXPLICIT LANGUAGE + COGNITIVE ROUNDTRIP",flush=True)
    print("="*78,flush=True)

    if not corpus.exists():
        raise SystemExit(f"BabyLM not found: {corpus}")
    if not db.exists():
        raise SystemExit(f"ConceptNet not found: {db}")

    print("[1/6] Loading ConceptNet...",flush=True)
    graph=IndexedConceptNet(db).build_index()
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

    print(
        f"      concepts={len(graph.concepts):,} "
        f"edges={graph.edge_count:,}",
        flush=True,
    )

    print("[2/6] Loading BabyLM...",flush=True)
    reader=BabyLMReader()
    lines=list(
        reader.lines(
            corpus,
            limit=args.train_limit+args.heldout,
        )
    )
    if not lines:
        raise SystemExit("BabyLM yielded no readable records.")
    train=lines[:-args.heldout] if args.heldout else lines
    heldout=lines[-args.heldout:] if args.heldout else []
    print(
        f"      train={len(train):,} heldout={len(heldout):,}",
        flush=True,
    )

    print("[3/6] Learning grammar...",flush=True)
    learner=GrammarCognitiveLearner(arch)
    for i,s in enumerate(train,1):
        learner.observe_sentence(s,learn=True)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(train)
        ):
            print(
                f"      train={i:,}/{len(train):,} "
                f"rules={len(learner.grammar.memory.rules):,} "
                f"observations={learner.grammar_observations:,}",
                flush=True,
            )

    print("[4/6] Explicit perception on arbitrary held-out input...",flush=True)
    bench=ExplicitRoundtrip(arch)

    tested=heldout[:args.max_cases]
    perception=[]
    for i,s in enumerate(tested,1):
        state=bench.interpreter.perceive(s)
        perception.append(state)
        if args.progress_every and (
            i%args.progress_every==0 or i==len(tested)
        ):
            print(
                f"      perceived={i:,}/{len(tested):,} "
                f"semantic_content="
                f"{sum(bool(x.entities) for x in perception):,}",
                flush=True,
            )

    print("[5/6] Bidirectional roundtrip...",flush=True)
    p2g=[
        bench.p2g2p(s)
        for s in tested
    ]
    g2p=[
        bench.g2p2g(st)
        for st in perception
        if st.entities
    ]

    p2g_acc=sum(int(x["pass"]) for x in p2g)/max(1,len(p2g))
    g2p_acc=sum(int(x["pass"]) for x in g2p)/max(1,len(g2p))

    explicit_coverage=sum(
        bool(x.entities or x.predicates)
        for x in perception
    )/max(1,len(perception))

    checks={
        "conceptnet_loaded":graph.edge_count>0,
        "babylm_loaded":bool(train and heldout),
        "grammar_learned":bool(learner.grammar.memory.rules),
        "explicit_states_produced":explicit_coverage>0,
        "p2g_cases":len(p2g)>0,
        "g2p_cases":len(g2p)>0,
        "p2g_pass":p2g_acc>=0.80,
        "g2p_pass":g2p_acc>=0.80,
    }

    print("[6/6] Final checks...",flush=True)
    for k,v in checks.items():
        print(
            f"  {k:30} {'PASS' if v else 'FAIL'}",
            flush=True,
        )

    status="PASS" if all(checks.values()) else "FAIL"

    report={
        "status":status,
        "version":"v393",
        "real_data":True,
        "conceptnet":{
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
        },
        "babylm":{
            "train_sentences":len(train),
            "heldout_sentences":len(heldout),
        },
        "grammar":{
            "rules":len(learner.grammar.memory.rules),
            "observations":learner.grammar_observations,
        },
        "explicit_representation":{
            "tested_sentences":len(tested),
            "perception_coverage":explicit_coverage,
            "states_with_predicates":sum(
                bool(x.predicates) for x in perception
            ),
            "states_with_arguments":sum(
                bool(x.arguments) for x in perception
            ),
            "states_with_grammar":sum(
                x.grammar.start_symbol=="S"
                for x in perception
            ),
        },
        "roundtrip":{
            "p2g_cases":len(p2g),
            "p2g_accuracy":p2g_acc,
            "g2p_cases":len(g2p),
            "g2p_accuracy":g2p_acc,
        },
        "checks":checks,
        "examples":{
            "p2g":[
                {
                    "pass":x["pass"],
                    "input":x["input_state"].tokens[0].text
                    if x["input_state"].tokens else None,
                    "generated":x["generated"],
                }
                for x in p2g[:5]
            ],
            "g2p":[
                {
                    "pass":x["pass"],
                    "generated":x["generated"],
                    "regenerated":x["regenerated"],
                }
                for x in g2p[:5]
            ],
        },
        "wall_time_seconds":time.perf_counter()-start,
    }

    out=Path.cwd()/"results"/"v393_explicit_language_benchmark.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(report,indent=2,default=str),
        encoding="utf-8",
    )

    print(f"[RESULT] {status}",flush=True)
    print(f"[RESULT FILE] {out.resolve()}",flush=True)
    graph.close()
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "corpus",nargs="?",type=Path,
        default=Path(r".\data\BabyLM-2026-Strict-Small"),
    )
    ap.add_argument(
        "--conceptnet",type=Path,
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
    real_run(args)


if __name__=="__main__":
    main()
