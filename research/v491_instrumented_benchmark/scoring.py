
from __future__ import annotations

import re
from dataclasses import dataclass,asdict


def norm(s):
    return re.sub(r"\s+"," ",str(s or "").strip().lower())


def words(s):
    return {
        x for x in re.findall(r"[a-z0-9']+",norm(s))
        if len(x)>1
    }


def naturalness(text):
    t=norm(text)
    if not t:return 0.0
    bad=(
        "i know:",
        "domain computer_support",
        "the user is",
        "the assistant is",
        "a useful candidate",
        "candidate proposition",
        "the architecture",
    )
    if any(x in t for x in bad):return 0.1
    if t.endswith((".","!","?")):return 1.0
    return 0.7


def brevity(text):
    n=len(str(text or "").split())
    if n==0:return 0.0
    if n<=8:return 1.0
    if n<=16:return 0.9
    if n<=30:return 0.75
    if n<=60:return 0.55
    return 0.3


@dataclass
class CaseScore:
    name:str
    passed:bool
    checks_passed:int
    checks_total:int
    check_rate:float
    naturalness:float
    state_consistency:float
    context_use:float
    non_parroting:float
    brevity:float
    failure_class:str
    details:dict
    responses:list[str]

    @property
    def overall(self):
        # Failed deterministic checks must actually lower the overall score.
        return (
            self.check_rate
            +self.naturalness
            +self.state_consistency
            +self.context_use
            +self.non_parroting
            +self.brevity
        )/6.0

    def to_dict(self):
        d=asdict(self)
        d["overall"]=self.overall
        return d


def classify_failure(case,details,responses):
    failed=[k for k,v in details.items() if not v]
    if not failed:
        return "none"

    name=case["name"]

    if name in {
        "dog_color","dog_count","state_update",
        "state_reference","topic_switch","followup_reference",
        "conversation_scope",
    }:
        return "state_or_reference"

    if name in {"addition","subtraction","letter_count"}:
        return "reasoning_or_symbolic"

    if name in {"spelling","example","colors"}:
        return "generation_or_language"

    if name in {"assistant_curiosity","assistant_wants","assistant_thoughts"}:
        return "assistant_state"

    if name in {"stale_response","followup_not_repeat","no_parrot"}:
        return "dialogue_control"

    return "general_response"


def check_case(case,responses):
    checks=case["checks"]
    final=norm(responses[-1] if responses else "")
    details={}

    if checks.get("natural"):
        details["natural"]=naturalness(final)>=0.7

    if checks.get("nonempty"):
        details["nonempty"]=bool(final)

    if "must_contain" in checks:
        details["must_contain"]=all(
            norm(x) in final for x in checks["must_contain"]
        )

    if "must_contain_one" in checks:
        details["must_contain_one"]=any(
            norm(x) in final for x in checks["must_contain_one"]
        )

    if "must_not_contain" in checks:
        details["must_not_contain"]=all(
            norm(x) not in final for x in checks["must_not_contain"]
        )

    if checks.get("not_exact_user"):
        details["not_exact_user"]=final!=norm(case["turns"][-1])

    if checks.get("varied"):
        details["varied"]=len({norm(x) for x in responses})>1

    if checks.get("count_at_least") is not None:
        colors=("red","blue","green","yellow","orange","purple","black","white","pink")
        details["count_at_least"]=sum(final.count(x) for x in colors)>=checks["count_at_least"]

    passed=sum(bool(v) for v in details.values())
    total=len(details)

    name=case["name"]
    state=1.0

    if name=="dog_color":
        state=1.0 if "red" in final and "computer_support" not in final else 0.0
    elif name=="dog_count":
        state=1.0 if ("two" in final or re.search(r"\b2\b",final)) else 0.0
    elif name=="state_update":
        state=1.0 if "blue" in final and "red" not in final else 0.0
    elif name=="topic_switch":
        state=1.0 if "red" in final and "universe" not in final else 0.0
    elif name=="letter_count":
        state=1.0 if ("3" in final or re.search(r"\bthree\b",final)) else 0.0

    context=1.0
    if len(responses)>1:
        context=1.0 if len({norm(x) for x in responses})==len(responses) else 0.5

    overlaps=[]
    for user,answer in zip(case["turns"],responses):
        a=words(answer)
        if a:
            overlaps.append(
                len(words(user)&a)/max(1,len(a))
            )

    nonparrot=1.0-max(overlaps or [0.0])

    return CaseScore(
        name=name,
        passed=(passed==total if total else True),
        checks_passed=passed,
        checks_total=total,
        check_rate=passed/max(1,total),
        naturalness=sum(naturalness(x) for x in responses)/max(1,len(responses)),
        state_consistency=state,
        context_use=context,
        non_parroting=nonparrot,
        brevity=sum(brevity(x) for x in responses)/max(1,len(responses)),
        failure_class=classify_failure(case,details,responses),
        details=details,
        responses=responses,
    )
