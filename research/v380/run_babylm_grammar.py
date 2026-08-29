
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def now():
    return time.perf_counter()


def log(message, start_time, **fields):
    elapsed=time.perf_counter()-start_time
    suffix=""
    if fields:
        suffix=" | " + " ".join(
            f"{k}={v}" for k,v in fields.items()
        )
    print(
        f"[{elapsed:8.2f}s] {message}{suffix}",
        flush=True,
    )


def write_result(result, output_path: Path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"[RESULT FILE] {output_path.resolve()}",
        flush=True,
    )


def main():
    p=argparse.ArgumentParser(
        description=(
            "Instrumented BabyLM + cognitive grammar learning run."
        )
    )
    p.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path(
            r".\data\BabyLM-2026-Strict-Small"
        ),
    )
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(
            r".\data\conceptnet_compact.db"
        ),
    )
    p.add_argument(
        "--train-limit",
        type=int,
        default=10000,
    )
    p.add_argument(
        "--heldout",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
    )
    p.add_argument(
        "--semantic-refresh-every",
        type=int,
        default=25,
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print Stage 6 progress every N training sentences.",
    )
    p.add_argument(
        "--result-name",
        default="v380_babylm_grammar.json",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
    )
    args=p.parse_args()

    from babylm_grammar import (
        GrammarCognitiveLearner,
        BabyLMReader,
    )

    start=now()
    results_dir=(
        Path.cwd() / "results"
    )
    result_path=results_dir / args.result_name

    print("=" * 78, flush=True)
    print("V380 INSTRUMENTED BABYLM + COGNITIVE GRAMMAR", flush=True)
    print("=" * 78, flush=True)
    log(
        "START",
        start,
        cwd=Path.cwd(),
    )

    if args.smoke:
        log("MODE=DATA_FREE_SMOKE",start)
        from babylm_grammar import smoke
        result=smoke()
        result["instrumentation"]={
            "mode":"smoke",
            "wall_time_seconds":time.perf_counter()-start,
        }
        write_result(
            result,
            result_path,
        )
        log("COMPLETE",start)
        return

    corpus=args.corpus.resolve()
    conceptnet=args.conceptnet.resolve()

    log("STAGE 1/8 — validating BabyLM path",start)
    if not corpus.exists():
        raise SystemExit(
            f"BabyLM dataset not found: {corpus}"
        )

    reader=BabyLMReader()
    files=reader.files(corpus)
    if not files:
        raise SystemExit(
            f"BabyLM dataset contains no supported files: {corpus}"
        )

    log(
        "BabyLM FOUND",
        start,
        path=corpus,
        files=len(files),
    )

    log("STAGE 2/8 — validating ConceptNet path",start)
    if not conceptnet.exists():
        raise SystemExit(
            f"ConceptNet database not found: {conceptnet}"
        )
    log(
        "ConceptNet FOUND",
        start,
        path=conceptnet,
        bytes=conceptnet.stat().st_size,
    )

    log(
        "STAGE 3/8 — loading/indexing semantic graph",
        start,
    )
    from real_grounding import IndexedConceptNet
    from semantic_memory import (
        IndexedSemanticMemory,
        SemanticEdge,
    )
    from semantic_architecture import (
        IntegratedSemanticArchitecture,
    )

    graph_start=now()
    graph=IndexedConceptNet(
        conceptnet
    ).build_index()
    graph_seconds=time.perf_counter()-graph_start

    log(
        "ConceptNet INDEX READY",
        start,
        concepts=f"{len(graph.concepts):,}",
        edges=f"{graph.edge_count:,}",
        stage_seconds=f"{graph_seconds:.2f}",
    )

    log(
        "STAGE 4/8 — preparing cognitive semantic substrate",
        start,
    )

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

    semantic=IntegratedSemanticArchitecture(
        memory
    )

    log(
        "COGNITIVE SEMANTIC SUBSTRATE READY",
        start,
        concepts=f"{len(graph.concepts):,}",
        edges=f"{graph.edge_count:,}",
    )

    log(
        "STAGE 5/8 — reading BabyLM records",
        start,
        requested_train=args.train_limit,
        heldout=args.heldout,
    )

    read_start=now()
    lines=list(
        reader.lines(
            corpus,
            limit=(
                args.train_limit
                + args.heldout
            ),
        )
    )
    read_seconds=time.perf_counter()-read_start

    if not lines:
        raise SystemExit(
            "BabyLM yielded no readable records."
        )

    train=(
        lines[:-args.heldout]
        if args.heldout
        else lines
    )
    heldout=(
        lines[-args.heldout:]
        if args.heldout
        else []
    )

    log(
        "BABYLM RECORDS READY",
        start,
        train=len(train),
        heldout=len(heldout),
        stage_seconds=f"{read_seconds:.2f}",
    )

    log(
        "STAGE 6/8 — incremental grammar learning",
        start,
    )

    learner=GrammarCognitiveLearner(
        semantic,
        semantic_refresh_every=(
            args.semantic_refresh_every
        ),
    )

    checkpoints=[]
    learning_start=now()

    def stage6_progress(i,total,learner):
        import time
        snap=learner.performance_snapshot()
        totals=snap["profile_totals_seconds"]

        processed=max(1,i)
        elapsed=time.perf_counter()-learning_start
        sps=processed/max(1e-9,elapsed)

        print(
            f"[{time.perf_counter()-start:8.2f}s] "
            f"STAGE 6 PROGRESS "
            f"| episode={i}/{total} "
            f"pct={100*i/max(1,total):5.1f}% "
            f"sent_per_s={sps:6.2f} "
            f"rules={len(learner.grammar.memory.rules)} "
            f"sem_calls={snap['semantic_requests']} "
            f"cache_hits={snap['semantic_cache_hits']} "
            f"cache_rate={snap['semantic_cache_hit_rate']:.3f}",
            flush=True,
        )

        grand=max(1e-9,totals["total_observe"])
        phase_lines=[
            ("tokenize",totals["tokenize"]),
            ("candidate_rules",totals["candidate_rules"]),
            ("semantic_total",totals["semantic_total"]),
            ("semantic_cognitive_call",totals["semantic_cognitive_call"]),
            ("hypothesis_scoring",totals["hypothesis_scoring"]),
            ("grammar_memory_update",totals["grammar_memory_update"]),
            ("posterior",totals["posterior"]),
        ]
        detail=" ".join(
            f"{name}={100*val/grand:4.1f}%"
            for name,val in phase_lines
        )
        print(
            f"[{time.perf_counter()-start:8.2f}s] "
            f"STAGE 6 PROFILE | {detail}",
            flush=True,
        )

        slow=learner.performance_snapshot()["slow_events"]
        if slow:
            latest=slow[-1]
            if latest["sentence_s"]>=0.25:
                print(
                    f"[{time.perf_counter()-start:8.2f}s] "
                    f"STAGE 6 LAST SLOW "
                    f"| total={latest['sentence_s']:.3f}s "
                    f"tokenize={latest['tokenize_s']:.3f}s "
                    f"rules={latest['candidate_rules_s']:.3f}s "
                    f"semantic={latest['semantic_s']:.3f}s "
                    f"score={latest['hypothesis_scoring_s']:.3f}s "
                    f"posterior={latest['posterior_s']:.3f}s",
                    flush=True,
                )

                for call in latest["semantic_calls"]:
                    if call["cognitive_call_s"]>=0.10:
                        print(
                            f"[{time.perf_counter()-start:8.2f}s] "
                            f"STAGE 6 SLOW SEMANTIC CALL "
                            f"| query={call['query']!r} "
                            f"seconds={call['cognitive_call_s']:.3f} "
                            f"cache_hit={call['cache_hit']}",
                            flush=True,
                        )

    for i,sentence in enumerate(train,1):
        learner.observe_sentence(
            sentence,
            learn=True,
        )

        if (
            args.progress_every
            and (
                i % args.progress_every==0
                or i==1
                or i==len(train)
            )
        ):
            stage6_progress(
                i,
                len(train),
                learner,
            )

        if (
            args.checkpoint_every
            and (
                i % args.checkpoint_every==0
                or i==len(train)
            )
        ):
            snapshot={
                "episode":i,
                **learner.grammar.memory.snapshot(),
                **learner.performance_snapshot(),
            }
            checkpoints.append(snapshot)

            log(
                "LEARNING CHECKPOINT",
                start,
                episode=i,
                rules=snapshot["rules"],
                tokens=snapshot["tokens"],
                semantic_requests=(
                    snapshot["semantic_requests"]
                ),
                cache_hits=(
                    snapshot["semantic_cache_hits"]
                ),
                commitments=(
                    snapshot["commitments"]
                ),
            )

    learning_seconds=(
        time.perf_counter()-learning_start
    )

    log(
        "GRAMMAR LEARNING COMPLETE",
        start,
        rules=len(learner.grammar.memory.rules),
        tokens=learner.grammar.memory.token_count,
        rejected=learner.grammar.memory.rejected_tokens,
        commits=learner.grammar.memory.commitments,
        stage_seconds=f"{learning_seconds:.2f}",
    )

    log(
        "STAGE 7/8 — held-out grammar evaluation",
        start,
        sentences=len(heldout),
    )

    heldout_start=now()
    heldout_metrics=learner.evaluate_heldout(
        heldout
    )
    heldout_seconds=time.perf_counter()-heldout_start

    log(
        "HELD-OUT EVALUATION COMPLETE",
        start,
        reusable_rule_hit=(
            f"{heldout_metrics['reusable_rule_hit_rate']:.3f}"
        ),
        commit_rate=(
            f"{heldout_metrics['commit_rate']:.3f}"
        ),
        stage_seconds=f"{heldout_seconds:.2f}",
    )

    log(
        "STAGE 8/8 — architecture/instrumentation checks",
        start,
    )

    instrumentation={
        "wall_time_seconds":(
            time.perf_counter()-start
        ),
        "stage_times_seconds":{
            "conceptnet_index":graph_seconds,
            "babylm_read":read_seconds,
            "grammar_learning":learning_seconds,
            "heldout_eval":heldout_seconds,
        },
        "semantic_performance":(
            learner.performance_snapshot()
        ),
        "stage6_instrumentation":{
            "progress_every":args.progress_every,
            "learning_seconds":learning_seconds,
            "sentences_per_second":(
                len(train)/max(1,learning_seconds)
            ),
        },
        "semantic_architecture_events":len(
            semantic.history
        ),
        "grammar_hypothesis_count":len(
            learner.grammar.memory.rules
        ),
        "cache_hit_rate":(
            learner.semantic_cache_hits
            / max(
                1,
                learner.semantic_requests
                + learner.semantic_cache_hits,
            )
        ),
    }

    checks={
        "babyLM_loaded":len(train)>0,
        "conceptnet_loaded":graph.edge_count>0,
        "grammar_nonempty":len(
            learner.grammar.memory.rules
        )>0,
        "semantic_events_nonempty":len(
            semantic.history
        )>0,
        "persistent_sentence_accounting":(
            learner.grammar.memory.sentence_count
            ==len(train)
        ),
        "heldout_present":len(heldout)>0,
        "cache_used":(
            learner.semantic_cache_hits>0
        ),
    }

    status="PASS" if all(checks.values()) else "FAIL"

    result={
        "status":status,
        "version":"v380-instrumented",
        "paths":{
            "corpus":str(corpus),
            "conceptnet":str(conceptnet),
            "result":str(
                result_path.resolve()
            ),
        },
        "conceptnet":{
            "concepts":len(graph.concepts),
            "edges":graph.edge_count,
            "index_seconds":graph_seconds,
        },
        "babylm":{
            "files":len(files),
            "train_sentences":len(train),
            "heldout_sentences":len(heldout),
        },
        "grammar":learner.grammar.memory.snapshot(),
        "heldout":heldout_metrics,
        "semantic_performance":learner.performance_snapshot(),
        "stage6_profile":learner.performance_snapshot(),
        "checks":checks,
        "instrumentation":instrumentation,
        "checkpoints":checkpoints,
    }

    print("CHECKS:")
    for name,value in checks.items():
        print(
            f"  {name:36} "
            f"{'PASS' if value else 'FAIL'}",
            flush=True,
        )

    print(
        f"[RESULT] {status}",
        flush=True,
    )

    write_result(
        result,
        result_path,
    )

    graph.close()


if __name__=="__main__":
    main()
