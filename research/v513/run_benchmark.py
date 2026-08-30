
from __future__ import annotations
import argparse,json,statistics,traceback
from pathlib import Path
from benchmark_cases import CASES
from benchmark_scorer import score_case
from architecture_adapters import ArchitectureAdapter,LLMOnlyAdapter

def run(name,adapter,verbose=False):
    rows=[]
    print(f"\n=== {name} ===",flush=True)
    for i,case in enumerate(CASES,1):
        adapter.reset();responses=[];diags=[]
        for turn in case["turns"]:
            try:
                answer=adapter.respond(turn)
                diag=adapter.diagnostics() or {}
            except Exception as exc:
                answer=""
                diag={"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(limit=3)}
            responses.append(str(answer or "").strip());diags.append(diag)
        result=score_case(case,responses,diags);rows.append(result)
        print(
            f"{i:02d}/{len(CASES)} {'PASS' if result.passed else 'FAIL':4} "
            f"{case['name']:<24} overall={result.overall:.3f} "
            f"stage={result.failure_stage or '-'}",
            flush=True,
        )
        if verbose and not result.passed:
            print("  U:",case["turns"][-1],flush=True)
            print("  A:",responses[-1],flush=True)
            print("  D:",json.dumps(diags[-1],ensure_ascii=False)[:3000],flush=True)
    return rows

def summarize(name,rows):
    stages={}
    for r in rows:
        if r.failure_stage:stages[r.failure_stage]=stages.get(r.failure_stage,0)+1
    return {
        "system":name,"cases":len(rows),
        "passed":sum(r.passed for r in rows),
        "pass_rate":sum(r.passed for r in rows)/len(rows),
        "overall":statistics.mean(r.overall for r in rows),
        "failure_stages":stages,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",required=True)
    ap.add_argument("--teacher",default="")
    ap.add_argument("--llm",action="store_true")
    ap.add_argument("--freeze-knowledge",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--output",default="v509_benchmark_results.json")
    args=ap.parse_args()

    summaries={};details={}

    arch=ArchitectureAdapter(
        args.memory,args.teacher,use_llm=False,trace=False,
        freeze=args.freeze_knowledge,
    )
    rows=run("Architecture alone",arch,args.verbose)
    summaries["architecture"]=summarize("Architecture alone",rows)
    details["architecture"]=[r.to_dict() for r in rows]

    if args.llm and args.teacher:
        combo=ArchitectureAdapter(
            args.memory,args.teacher,use_llm=True,trace=False,
            freeze=args.freeze_knowledge,
        )
        rows=run("Architecture + LLM",combo,args.verbose)
        summaries["architecture_llm"]=summarize("Architecture + LLM",rows)
        details["architecture_llm"]=[r.to_dict() for r in rows]

        llm=LLMOnlyAdapter(args.teacher)
        rows=run("LLM alone",llm,args.verbose)
        summaries["llm"]=summarize("LLM alone",rows)
        details["llm"]=[r.to_dict() for r in rows]

    print("\n=== SUMMARY ===")
    for item in summaries.values():
        print(
            f"{item['system']:<24}"
            f"pass={item['pass_rate']*100:5.1f}% "
            f"overall={item['overall']:.3f} "
            f"failures={item['failure_stages']}",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps({
            "benchmark":"v509_small_chat_diagnostic",
            "summaries":summaries,
            "cases":details,
        },indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    print("Results:",args.output)

if __name__=="__main__":
    main()
