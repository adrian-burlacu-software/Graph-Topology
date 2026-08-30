from __future__ import annotations
import argparse, json, statistics, traceback
from pathlib import Path

from benchmark_cases import CASES as V509_CASES
from benchmark_v510_cases import V510_ADVERSARIAL_CASES
from benchmark_v511_cases import SWITCH_PROBES
from benchmark_scorer import score_case
from architecture_adapters import ArchitectureAdapter


def normalized_case(case):
    c=dict(case)
    checks=dict(c.get("checks",{}))
    if "must_contain" in checks and isinstance(checks["must_contain"],str):
        checks["must_contain"]=[checks["must_contain"]]
    c["checks"]=checks
    return c


def run_suite(name,cases,adapter,verbose=False):
    rows=[]
    print(f"\n=== {name} ({len(cases)} cases) ===",flush=True)
    for i,raw in enumerate(cases,1):
        case=normalized_case(raw)
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
        row=result.to_dict()
        row["attention_trajectory"]=[d.get("attention",{}) for d in diags]
        row["turn_diagnostics"]=diags
        rows.append(row)
        focus=case.get("focus","regression")
        print(f"{i:03d}/{len(cases)} {'PASS' if result.passed else 'FAIL':4} {case['name']:<32} focus={focus:<22} stage={result.failure_stage or '-'}",flush=True)
        if verbose:
            print("  attention:",json.dumps(row["attention_trajectory"],ensure_ascii=False)[:5000],flush=True)
    return rows


def attention_stats(rows):
    events=[]
    for r in rows:
        traj=r.get("attention_trajectory",[])
        for i,a in enumerate(traj):
            if not a: continue
            events.append(a)
    switches=[a for a in events if a.get("target_changed")]
    stable=[a for a in events if not a.get("target_changed")]
    switch_costs=[a.get("switch_cost_proxy",0) for a in switches]
    margins=[a.get("planner",{}).get("margin") for a in events if a.get("planner",{}).get("margin") is not None]
    return {
        "turns_instrumented":len(events),
        "target_switches":len(switches),
        "stable_turns":len(stable),
        "switch_cost_proxy_mean":statistics.mean(switch_costs) if switch_costs else 0.0,
        "switch_cost_proxy_sum":sum(switch_costs),
        "planner_margin_mean":statistics.mean(margins) if margins else 0.0,
        "selected_sources":{
            s:sum(1 for a in events if a.get("selected_source")==s)
            for s in sorted({a.get("selected_source") for a in events if a.get("selected_source")})
        },
    }


def summarize(name,rows):
    stages={}
    for r in rows:
        if r.get("failure_stage"):
            stages[r["failure_stage"]]=stages.get(r["failure_stage"],0)+1
    return {
        "system":name,
        "cases":len(rows),
        "passed":sum(bool(r.get("passed")) for r in rows),
        "pass_rate":sum(bool(r.get("passed")) for r in rows)/len(rows) if rows else 0.0,
        "overall":statistics.mean(r.get("overall",0.0) for r in rows) if rows else 0.0,
        "failure_stages":stages,
        "attention":attention_stats(rows),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",required=True)
    ap.add_argument("--freeze-knowledge",action="store_true")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--output",default="v511_benchmark_results.json")
    args=ap.parse_args()

    adapter=ArchitectureAdapter(args.memory,use_llm=False,trace=False,freeze=args.freeze_knowledge)
    regression=run_suite("V509 regression",V509_CASES,adapter,args.verbose)
    adversarial=run_suite("V510 adversarial",V510_ADVERSARIAL_CASES,adapter,args.verbose)
    probes=run_suite("V511 switch probes",SWITCH_PROBES,adapter,args.verbose)
    all_rows=regression+adversarial
    summaries={
        "v509_regression":summarize("V509 regression",regression),
        "v510_adversarial":summarize("V510 adversarial",adversarial),
        "combined":summarize("Combined",all_rows),
        "v511_switch_probes":summarize("V511 switch probes",probes),
    }
    print("\n=== SUMMARY ===",flush=True)
    for item in summaries.values():
        print(f"{item['system']:<22} pass={item['pass_rate']*100:5.1f}% overall={item['overall']:.3f} failures={item['failure_stages']} switches={item['attention']['target_switches']} switch_cost={item['attention']['switch_cost_proxy_sum']}",flush=True)

    payload={
        "benchmark":"v511_attention_trajectory",
        "description":"V509 regression + V510 adversarial cases with planner/attention trajectory instrumentation, plus explicit target-switch probes.",
        "knowledge_policy":"frozen" if args.freeze_knowledge else "not_frozen",
        "suites":summaries,
        "cases":{
            "v509_regression":regression,
            "v510_adversarial":adversarial,
            "v511_switch_probes":probes,
        },
    }
    out=Path(args.output)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Results:",out,flush=True)

if __name__=="__main__":
    main()
