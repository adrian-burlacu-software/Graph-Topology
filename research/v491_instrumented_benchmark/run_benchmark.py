
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark_cases import CASES
from scoring import check_case
from adapters import LLMOnlyAdapter,load_factory
from architecture_core import Architecture
from architecture_llm_adapter import Adapter as ArchitectureLLMAdapter


class NativeArchitectureAdapter:
    name="architecture"
    def __init__(self):
        self.engine=Architecture()
    def reset(self):
        self.engine.reset()
    def respond(self,text):
        return self.engine.respond(text)


def run(name,adapter,verbose=False):
    print(f"\n=== {name} ===")
    scores=[]

    for i,case in enumerate(CASES,1):
        adapter.reset()
        responses=[]

        for turn_index,turn in enumerate(case["turns"],1):
            try:
                response=str(
                    adapter.respond(turn) or ""
                ).strip()
            except Exception as exc:
                response=(
                    f"[ERROR] {type(exc).__name__}: {exc}"
                )
            responses.append(response)

            if verbose:
                print(
                    f"  [TURN {turn_index}] U={turn!r}",
                    flush=True,
                )
                print(
                    f"  [TURN {turn_index}] A={response!r}",
                    flush=True,
                )

        score=check_case(case,responses)
        scores.append(score)

        status="PASS" if score.passed else "FAIL"
        print(
            f"{i:02d}/{len(CASES)} {status:<4} "
            f"{case['name']:<24} "
            f"check={score.check_rate:.2f} "
            f"overall={score.overall:.3f} "
            f"state={score.state_consistency:.2f} "
            f"context={score.context_use:.2f} "
            f"natural={score.naturalness:.2f} "
            f"nonparrot={score.non_parroting:.2f} "
            f"brief={score.brevity:.2f}"
        )

        if not score.passed:
            print(
                f"      [FAILURE] class={score.failure_class} "
                f"checks={score.details}",
                flush=True,
            )
            print(
                f"      [FINAL] {responses[-1]!r}",
                flush=True,
            )

    return scores


def summarize(name,scores):
    return {
        "system":name,
        "cases":len(scores),
        "passed":sum(s.passed for s in scores),
        "pass_rate":sum(s.passed for s in scores)/max(1,len(scores)),
        "overall":statistics.mean(s.overall for s in scores),
        "check_rate":statistics.mean(s.check_rate for s in scores),
        "naturalness":statistics.mean(s.naturalness for s in scores),
        "state_consistency":statistics.mean(s.state_consistency for s in scores),
        "context_use":statistics.mean(s.context_use for s in scores),
        "non_parroting":statistics.mean(s.non_parroting for s in scores),
        "brevity":statistics.mean(s.brevity for s in scores),
        "failures":{
            k:sum(
                s.failure_class==k
                for s in scores
            )
            for k in sorted({
                s.failure_class for s in scores
                if s.failure_class!="none"
            })
        },
    }


def print_summary(summaries):
    print("\n=== SUMMARY ===")
    print(
        f"{'SYSTEM':<24}{'PASS':>9}{'CHECK':>9}"
        f"{'OVERALL':>10}{'STATE':>10}{'CONTEXT':>10}"
        f"{'NATURAL':>10}{'NONPARROT':>12}{'BRIEF':>10}"
    )
    for x in summaries.values():
        print(
            f"{x['system']:<24}"
            f"{x['pass_rate']*100:>8.1f}%"
            f"{x['check_rate']:>9.3f}"
            f"{x['overall']:>10.3f}"
            f"{x['state_consistency']:>10.3f}"
            f"{x['context_use']:>10.3f}"
            f"{x['naturalness']:>10.3f}"
            f"{x['non_parroting']:>12.3f}"
            f"{x['brevity']:>10.3f}"
        )
        if x["failures"]:
            print(
                f"  failure_classes={x['failures']}"
            )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",default="")
    ap.add_argument("--architecture-factory",default="")
    ap.add_argument("--architecture-llm-model",default="")
    ap.add_argument("--output",default="benchmark_results.json")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--list-cases",action="store_true")
    args=ap.parse_args()

    if args.list_cases:
        for i,c in enumerate(CASES,1):
            print(
                f"{i:02d}. {c['name']}: "
                f"{' | '.join(c['turns'])}"
            )
        return

    summaries={}
    cases={}

    if args.teacher:
        scores=run(
            "LLM alone",
            LLMOnlyAdapter(args.teacher),
            args.verbose,
        )
        summaries["llm"]=summarize("LLM alone",scores)
        cases["llm"]=[x.to_dict() for x in scores]

    if args.architecture_factory:
        factory=load_factory(args.architecture_factory)
        scores=run(
            "Architecture alone",
            factory(name="architecture"),
            args.verbose,
        )
        summaries["architecture"]=summarize(
            "Architecture alone",scores
        )
        cases["architecture"]=[
            x.to_dict() for x in scores
        ]

    if args.architecture_llm_model:
        scores=run(
            "Architecture + LLM",
            ArchitectureLLMAdapter(
                args.architecture_llm_model
            ),
            args.verbose,
        )
        summaries["architecture_llm"]=summarize(
            "Architecture + LLM",scores
        )
        cases["architecture_llm"]=[
            x.to_dict() for x in scores
        ]

    print_summary(summaries)

    Path(args.output).write_text(
        json.dumps(
            {
                "summary":summaries,
                "cases":cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults: {args.output}")


if __name__=="__main__":
    main()
