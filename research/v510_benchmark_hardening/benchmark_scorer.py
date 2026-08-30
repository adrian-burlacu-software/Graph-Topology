
from __future__ import annotations
from dataclasses import dataclass,asdict
import re

def norm(x): return re.sub(r"\s+"," ",str(x or "").strip().lower())
def natural(x):
    t=norm(x)
    if not t:return 0.0
    bad=("computer_support","i know:","candidate proposition","the architecture","response role")
    if any(x in t for x in bad):return .1
    return 1.0 if t.endswith((".","!","?")) else .7
def brief(x):
    n=len(norm(x).split())
    return 0 if n==0 else 1 if n<=8 else .9 if n<=16 else .75 if n<=30 else .55

@dataclass
class CaseResult:
    name:str
    passed:bool
    checks:dict
    overall:float
    failure_stage:str|None
    diagnostic:dict
    responses:list[str]
    def to_dict(self): return asdict(self)

def failure_stage(case,d,responses):
    if d.get("error"): return "exception"
    last=case["turns"][-1].lower()
    if any(x in last for x in ("how many","what if","how do you spell","r's","one plus one")) and not d.get("logic_answer"):
        return "logic/operator"
    if any(x in last for x in ("what color","what colour","is it","what is","why")) and d.get("target")=="general":
        return "target/reference"
    if d.get("participant_consulted") and d.get("participant_accepted") is False:
        return "participant/evaluation"
    if d.get("participant_consulted") and d.get("source")=="participant":
        return "participant/selection"
    if any(x in norm(responses[-1] if responses else "") for x in ("computer_support","candidate proposition","i know:")):
        return "surface/output"
    return "goal/planner"

def score_case(case,responses,diags):
    final=norm(responses[-1] if responses else "")
    c=case.get("checks",{})
    checks={}
    if c.get("natural"): checks["natural"]=natural(final)>=.7
    if c.get("nonempty"): checks["nonempty"]=bool(final)
    if "must_contain" in c:checks["must_contain"]=all(norm(x) in final for x in c["must_contain"])
    if "must_contain_one" in c:checks["must_contain_one"]=any(norm(x) in final for x in c["must_contain_one"])
    if "must_not_contain" in c:checks["must_not_contain"]=all(norm(x) not in final for x in c["must_not_contain"])
    if c.get("not_exact_user"):checks["not_exact_user"]=final!=norm(case["turns"][-1])
    if c.get("varied"):checks["varied"]=len({norm(x) for x in responses})>1
    passed=all(checks.values()) if checks else bool(final)
    overall=(natural(final)+(1.0 if final else 0.0)+(0.0 if any(x in final for x in ("i know:","computer_support","candidate proposition")) else 1.0)+brief(final))/4
    return CaseResult(
        case["name"],passed,checks,overall,
        None if passed else failure_stage(case,diags[-1] if diags else {},responses),
        diags[-1] if diags else {},
        responses,
    )
