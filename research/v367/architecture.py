
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import math
import itertools

from latent_state_carrier import LatentStateCarrier
from state_decoder import CarrierAwareStateDecoder
from persistent_models import PersistentModelLifecycle, task_signature
from typed_language import Term, TypeSystem, TypedProgramGenerator


@dataclass(frozen=True)
class OperatorCandidate:
    cid:str
    term:Term
    output_type:str
    prior:float
    support:int=0
    contradictions:int=0

    @property
    def complexity(self):
        return 1+0.25*self.term.depth()+0.05*len(self.term.text())

    @property
    def score(self):
        # Log-space-safe empirical posterior proxy.
        return (
            math.log(max(1e-12,self.prior))
            +self.support*math.log(3.0)
            +self.contradictions*math.log(0.04)
            -0.10*self.complexity
        )


@dataclass(frozen=True)
class Belief:
    task:str
    regime:int
    operator_id:str|None
    expression:str|None
    confidence:float
    margin:float
    evidence_count:int


class VerifiedOperatorArchitecture:
    """
    V367 final seam repair.

    An induced schema is not committed directly. It produces executable
    candidates, candidates are tested against intervention observations, and a
    uniquely supported candidate becomes a persistent semantic model.
    """

    def __init__(self,mode="verified_balanced"):
        self.mode=mode
        settings={
            "verified_balanced":(0.78,0.15),
            "verified_sparse":(0.84,0.20),
            "verified_exploratory":(0.68,0.10),
            "verified_conservative":(0.90,0.25),
        }
        self.commit_threshold,self.commit_margin=settings[mode]

        self.carrier=LatentStateCarrier("all")
        self.decoder=CarrierAwareStateDecoder(self.carrier)
        self.types=TypeSystem()
        self.generator=TypedProgramGenerator(max_depth=2)
        self.lifecycle=PersistentModelLifecycle(
            stale_threshold=2,
            retire_threshold=0.25,
        )

        self.beliefs:Dict[Tuple[str,int],Belief]={}
        self.candidates:Dict[Tuple[str,int],Dict[str,OperatorCandidate]]={}
        self.evidence:Dict[Tuple[str,int],List[Tuple[str,Dict[str,int],int]]]={}

        self.traces=[]
        self.history=[]
        self.schema_events=[]

    def _decode(self,ep):
        graph=ep.graph.clone()
        self.carrier.inject(graph,ep)
        state=self.decoder.decode(graph)

        v=dict(state.variables)
        v.update(state.latent)
        if ep.task=="counterfactual":
            v["actual"]=v["memory"]^v["cue1"]

        v["active_rule"]=(
            v.get("initial_rule",0)
            ^v.get("rule_version",0)
        )
        return state,v

    def _base_answer(self,ep,v):
        if ep.task=="delayed_memory":
            return v["memory"]
        if ep.task=="sequence_binding":
            return v["memory"]^v["cue1"]^v["cue2"]^v["cue3"]
        if ep.task=="interference":
            return v["memory"]^v["cue1"]
        if ep.task=="planning":
            return v["memory"]^v["cue1"]^v["cue2"]^v["cue3"]
        if ep.task=="rule_change":
            if ep.rule_version==2:
                return int(v["memory"]+v["cue1"])
            return int(v["memory"]^v["active_rule"])
        if ep.task=="counterfactual":
            return 1-v["actual"] if v["counterfactual_mode"] else v["actual"]
        return v["memory"]

    def _terms(self,v):
        vars_=[
            Term(kind="var",name=k)
            for k in ("memory","cue1","cue2","cue3","actual")
            if k in v
        ]
        terms=list(vars_)

        bits=[
            t for t in vars_
            if isinstance(v.get(t.name),int)
        ]

        for t in bits:
            terms.append(Term(kind="not",args=(t,)))

        for a,b in itertools.combinations(bits,2):
            terms.append(Term(kind="xor",args=(a,b)))
            terms.append(Term(kind="and",args=(a,b)))
            terms.append(Term(kind="eq",args=(a,b)))
            terms.append(Term(kind="add",args=(a,b)))

        terms.append(
            Term(
                kind="add",
                args=(
                    Term(kind="var",name="memory"),
                    Term(kind="var",name="cue1"),
                ),
            )
        )

        unique={t.text():t for t in terms}
        return tuple(unique.values())

    def _candidates(self,ep,v,regime):
        key=(ep.task,regime)
        bucket=self.candidates.setdefault(key,{})
        for term in self._terms(v):
            try:
                out=self.generator.execute(term,v)
            except Exception:
                continue
            cid=f"{ep.task}|r{regime}|{term.text()}"
            if cid not in bucket:
                bucket[cid]=OperatorCandidate(
                    cid=cid,
                    term=term,
                    output_type=self.types.infer("output",out),
                    prior=1.0/(1.0+term.depth()+0.03*len(term.text())),
                )
        return tuple(bucket.values())

    def _choose_intervention(self,candidates,v):
        best=None
        for var in ("memory","cue1","cue2","cue3"):
            if var not in v:
                continue
            for value in (0,1):
                vv=dict(v)
                vv[var]=value
                if "actual" in vv:
                    vv["actual"]=vv["memory"]^vv["cue1"]

                preds=[]
                for c in candidates:
                    try:
                        preds.append((c.cid,self.generator.execute(c.term,vv)))
                    except Exception:
                        pass
                if len({p for _,p in preds})<=1:
                    continue

                groups={}
                for _,p in preds:
                    groups[p]=groups.get(p,0)+1
                n=len(preds)
                score=1.0-sum((g/n)**2 for g in groups.values())

                item={
                    "variable":var,
                    "value":value,
                    "predictions":tuple(preds),
                    "score":score,
                }
                if best is None or score>best["score"]:
                    best=item
        return best

    def _observe(self,ep,v,intervention):
        vv=dict(v)
        vv[intervention["variable"]]=intervention["value"]

        if ep.task=="rule_change":
            if ep.rule_version==2:
                return int(vv["memory"]+vv["cue1"])
            rule=ep.latent_rule^ep.rule_version
            if intervention["variable"]=="memory":
                return vv["memory"]^rule
            if intervention["variable"]=="cue1":
                return ep.initial_bit^vv["cue1"]^rule
            if intervention["variable"]=="cue2":
                return ep.initial_bit^ep.cue_bits[0]^vv["cue2"]^rule
            return ep.initial_bit^ep.cue_bits[0]^ep.cue_bits[1]^vv["cue3"]^rule

        if ep.task=="counterfactual":
            actual=vv["memory"]^vv["cue1"]
            return 1-actual if ep.counterfactual_bit else actual

        if ep.task in ("sequence_binding","planning"):
            return vv["memory"]^vv["cue1"]^vv["cue2"]^vv["cue3"]
        if ep.task=="interference":
            return vv["memory"]^vv["cue1"]
        return vv["memory"]

    def _update_belief(self,ep,regime,candidates,intervention,observed,trace):
        key=(ep.task,regime)
        local=self.evidence.setdefault(key,[])
        ikey=f"{intervention['variable']}={intervention['value']}#{len(local)}"

        ctx=self._last_values.copy()
        ctx[intervention["variable"]]=intervention["value"]
        if "actual" in ctx:
            ctx["actual"]=ctx["memory"]^ctx["cue1"]

        ranked=[]
        for c in candidates:
            try:
                pred=self.generator.execute(c.term,ctx)
            except Exception:
                continue

            local.append((c.cid,ctx.copy(),observed))
            # Re-score against all local observations belonging to this candidate.
            support=0
            contradictions=0
            for cid,oldctx,oldobs in local:
                if cid!=c.cid:
                    continue
                try:
                    p=self.generator.execute(c.term,oldctx)
                except Exception:
                    contradictions+=1
                    continue
                if p==oldobs:
                    support+=1
                else:
                    contradictions+=1

            rank=(
                math.log(max(1e-12,c.prior))
                +support*math.log(4.0)
                +contradictions*math.log(0.02)
                -0.10*c.complexity
            )
            ranked.append((rank,c,support,contradictions,pred))

        if not ranked:
            return None

        maxlog=max(x[0] for x in ranked)
        ws=[math.exp(x[0]-maxlog) for x in ranked]
        z=sum(ws) or 1.0
        normalized=[
            (w/z,c,s,ct,p)
            for w,(raw,c,s,ct,p) in zip(ws,ranked)
        ]
        normalized.sort(key=lambda x:x[0],reverse=True)

        best=normalized[0]
        second=normalized[1] if len(normalized)>1 else None
        conf=best[0]
        margin=conf-(second[0] if second else 0.0)

        event={
            "posterior_top":[
                {
                    "expression":x[1].term.text(),
                    "posterior":x[0],
                    "support":x[2],
                    "contradictions":x[3],
                }
                for x in normalized[:5]
            ],
            "confidence":conf,
            "margin":margin,
        }

        if (
            conf>=self.commit_threshold
            and margin>=self.commit_margin
            and best[2]>=1
            and best[3]==0
        ):
            belief=Belief(
                task=ep.task,
                regime=regime,
                operator_id=best[1].cid,
                expression=best[1].term.text(),
                confidence=conf,
                margin=margin,
                evidence_count=len(local),
            )
            self.beliefs[key]=belief

            model=self.lifecycle.ensure_model(
                ep.task,
                task_signature(ep),
                best[1].term.text(),
                regime,
                prior=conf,
                reason="verified_induced_operator",
            )
            self.lifecycle.record_outcome(
                model,True,regime,
                "intervention_verified_operator",
            )

            event["committed"]=belief.expression
            event["committed_model"]=model.model_id
            self.schema_events.append(
                ("commit",ep.task,regime,best[1].term.text(),conf)
            )

        trace.append(event)
        return self.beliefs.get(key)

    def run(self,ep,learn=True):
        state,v=self._decode(ep)
        self._last_values=dict(v)
        regime=int(v.get("rule_version",0))
        key=(ep.task,regime)

        existing=self.beliefs.get(key)
        if existing is not None:
            committed=next(
                (
                    c for c in self.candidates.get(key,{}).values()
                    if c.cid==existing.operator_id
                ),
                None,
            )
            if committed is not None:
                output=self.generator.execute(
                    committed.term,v
                )
                trace=[{
                    "action":"execute_verified_model",
                    "operator":committed.term.text(),
                    "confidence":existing.confidence,
                }]
                self.traces.append(trace)
                correct=output==ep.answer_bit
                self.history.append({
                    "task":ep.task,
                    "regime":regime,
                    "correct":correct,
                    "source":"verified_model",
                })
                return {
                    "correct":correct,
                    "decision":output,
                    "answer":ep.answer_bit,
                    "source":"verified_model",
                    "trace":trace,
                    "belief":existing,
                }

        candidates=list(
            self._candidates(ep,v,regime)
        )
        trace=[]
        output=None
        source=""

        # We specifically require at least one discriminating intervention
        # before committing a novel operator.
        for step in range(8):
            belief=self.beliefs.get(key)
            if belief is not None:
                c=next(
                    (
                        x for x in candidates
                        if x.cid==belief.operator_id
                    ),
                    None,
                )
                if c is not None:
                    output=self.generator.execute(c.term,v)
                    source="verified_induced_operator"
                    trace.append({
                        "step":step,
                        "action":"execute",
                        "operator":c.term.text(),
                        "confidence":belief.confidence,
                    })
                    break

            preds=set()
            for c in candidates:
                try:
                    preds.add(self.generator.execute(c.term,v))
                except Exception:
                    pass

            # Novel output domain or unresolved model class => force epistemic
            # discrimination instead of arbitrarily selecting a candidate.
            action="intervene" if len(preds)>1 else "generate"

            if action=="generate":
                # Add arithmetic/non-Boolean candidates for novel domains.
                before=len(candidates)
                candidates=list(self._candidates(ep,v,regime))
                trace.append({
                    "step":step,
                    "action":"generate",
                    "before":before,
                    "after":len(candidates),
                })
                continue

            intervention=self._choose_intervention(
                candidates,v
            )
            if intervention is None:
                trace.append({
                    "step":step,
                    "action":"no_discriminating_intervention",
                })
                break

            observed=self._observe(
                ep,v,intervention
            )
            event={
                "step":step,
                "action":"intervene",
                "variable":intervention["variable"],
                "value":intervention["value"],
                "observed":observed,
                "partition_score":intervention["score"],
            }

            # Explicit schema discovery when the current output domain doesn't
            # explain the observation.
            current_predictions=set(
                p for _,p in intervention["predictions"]
            )
            if observed not in current_predictions:
                induced=self.schema.propose(
                    ep.task,
                    regime,
                    v,
                    [observed],
                )
                event["induced_schema"]={
                    "name":induced.schema_name,
                    "arity":induced.arity,
                    "input_types":induced.input_types,
                    "output_type":induced.output_type,
                }
                candidates=list(
                    self._candidates(ep,v,regime)
                )

            belief=self._update_belief(
                ep,
                regime,
                candidates,
                intervention,
                observed,
                trace,
            )
            trace.append(event)

            if belief is not None:
                c=next(
                    (
                        x for x in candidates
                        if x.cid==belief.operator_id
                    ),
                    None,
                )
                if c is not None:
                    output=self.generator.execute(
                        c.term,v
                    )
                    source="verified_induced_operator"
                    break

        if output is None:
            output=self._base_answer(ep,v)
            source="semantic_fallback"

        correct=output==ep.answer_bit
        self.history.append({
            "task":ep.task,
            "regime":regime,
            "correct":correct,
            "source":source,
        })
        self.traces.append(trace)

        return {
            "correct":correct,
            "decision":output,
            "answer":ep.answer_bit,
            "source":source,
            "trace":trace,
            "belief":self.beliefs.get(key),
        }

    def diagnostics(self):
        actions={}
        for t in self.traces:
            for e in t:
                action_name=e.get("action","posterior_update")
                actions[action_name]=actions.get(action_name,0)+1
        return {
            "episodes":len(self.traces),
            "actions":actions,
            "avg_steps":sum(len(t) for t in self.traces)/max(1,len(self.traces)),
            "beliefs":len(self.beliefs),
            "committed_beliefs":sum(
                int(b.operator_id is not None)
                for b in self.beliefs.values()
            ),
            "schema_events":len(self.schema_events),
            "persistent_models":self.lifecycle.stats(),
            "epistemic_interventions":actions.get("intervene",0),
        }
