from __future__ import annotations
import argparse, json, statistics, traceback
from pathlib import Path

from benchmark_cases import CASES as V509_CASES
from benchmark_v510_cases import V510_ADVERSARIAL_CASES
from benchmark_scorer import score_case
from architecture_adapters import ArchitectureAdapter


def normalized_case(case):
    c=dict(case)
    checks=dict(c.get("checks",{}))
    if "must_contain" in checks and isinstance(checks["must_contain"],str):
        checks["must_contain"]=[checks["must_contain"]]
    c["checks"]=checks
    return c


def run_suite(name, cases, adapter, verbose=False):
    rows=[]
    print(f"\n=== {name} ({len(cases)} cases) ===", flush=True)
    for i,raw_case in enumerate(cases,1):
        case=normalized_case(raw_case)
        adapter.reset(); responses=[]; diags=[]
        for turn in case["turns"]:
            try:
                answer=adapter.respond(turn)
                diag=adapter.diagnostics() or {}
            except Exception as exc:
                answer=""
                diag={"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(limit=5)}
            responses.append(str(answer or "").strip()); diags.append(diag)
        result=score_case(case,responses,diags)
        rows.append(result)
        focus=case.get("focus", "regression")
        print(
            f"{i:03d}/{len(cases)} {'PASS' if result.passed else 'FAIL':4} "
            f"{case['name']:<32} focus={focus:<18} "
            f"stage={result.failure_stage or '-'}",
            flush=True,
        )
        if verbose and not result.passed:
            print("  U:",case["turns"][-1],flush=True)
            print("  A:",responses[-1],flush=True)
            print("  D:",json.dumps(diags[-1],ensure_ascii=False)[:3500],flush=True)
    return rows


def summarize(name, rows):
    stages={}
    for r in rows:
        if r.failure_stage:
            stages[r.failure_stage]=stages.get(r.failure_stage,0)+1
    return {
        "system":name,
        "cases":len(rows),
        "passed":sum(r.passed for r in rows),
        "pass_rate":sum(r.passed for r in rows)/len(rows),
        "overall":statistics.mean(r.overall for r in rows),
        "failure_stages":stages,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",required=True)
    ap.add_argument("--freeze-knowledge",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--output",default="v510_benchmark_results.json")
    args=ap.parse_args()

    adapter=ArchitectureAdapter(
        args.memory, use_llm=False, trace=False, freeze=args.freeze_knowledge
    )

    regression=run_suite("V509 regression",V509_CASES,adapter,args.verbose)
    adversarial=run_suite("V510 adversarial",V510_ADVERSARIAL_CASES,adapter,args.verbose)
    all_rows=regression+adversarial

    summaries={
        "v509_regression":summarize("V509 regression",regression),
        "v510_adversarial":summarize("V510 adversarial",adversarial),
        "combined":summarize("Combined",all_rows),
    }

    print("\n=== SUMMARY ===")
    for item in summaries.values():
        print(
            f"{item['system']:<20} "
            f"pass={item['pass_rate']*100:5.1f}% "
            f"overall={item['overall']:.3f} "
            f"failures={item['failure_stages']}",
            flush=True,
        )

    payload={
        "benchmark":"v510_benchmark_hardening",
        "description":"V509 regression plus adversarial semantic-boundary cases.",
        "suites":summaries,
        "cases":{
            "v509_regression":[r.to_dict() for r in regression],
            "v510_adversarial":[r.to_dict() for r in adversarial],
        },
    }
    Path(args.output).write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Results:",args.output)


if __name__=="__main__":
    main()
