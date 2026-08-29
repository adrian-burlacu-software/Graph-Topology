
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


class CreditBase:
    name="none"

    def reset(self):
        return None

    def inject_state(self,graph):
        pass

    def modify_decision(self,graph,decision,path):
        return int(decision)

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        return None


class NoneCredit(CreditBase):
    name="none"

    def __init__(self):
        self.count=0

    def feedback(
        self,
        graph,
        predicted,
        answer,
        path,
        episode,
    ):
        self.count+=1


class GlobalReward(CreditBase):
    """
    Global neuromodulatory signal.

    Improvement hypothesis:
    feedback produces a durable system-level learning signal that the planner
    can actually consume on the next trial.
    """
    name="global_reward"

    def __init__(self):
        self.signal=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "credit_state",
            "credit_state",
            value=self.signal,
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        # Positive signal means "base policy was probably wrong before";
        # this architecture uses it as a learned policy inversion.
        if self.signal>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        error=float(
            predicted!=answer
        )
        if error:
            self.signal=1.0
        self.count+=1


class Eligibility(CreditBase):
    """
    Temporal eligibility trace.

    Improvement hypothesis:
    keep an eligibility state that survives the delay and only convert
    recent evidence into a durable policy signal gradually.
    """
    name="eligibility"

    def __init__(self):
        self.trace=0.0
        self.signal=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "eligibility_state",
            "eligibility_state",
            value=self.trace,
            persistent=True,
        )
        graph.add_node(
            "credit_state",
            "credit_state",
            value=self.signal,
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        if self.signal>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        error=float(
            predicted!=answer
        )

        self.trace=(
            0.70*self.trace
            +0.30*error
        )

        self.signal=max(
            self.signal,
            self.trace,
        )

        self.count+=1


class LocalPathCredit(CreditBase):
    """
    Assign credit to the actual recalled route.

    Improvement hypothesis:
    instead of a global policy scalar, reinforce/inhibit the structural path
    that caused the decision.
    """
    name="local_path"

    def __init__(self):
        self.edge_credit:Dict[
            Tuple[str,str,str],
            float,
        ]=defaultdict(float)
        self.count=0

    def inject_state(self,graph):
        # Graph-native persistence is the point of this candidate.
        for key,value in list(
            self.edge_credit.items()
        ):
            src,rel,dst=key
            graph.add_edge(
                src,
                "credit",
                dst,
                weight=value,
                persistent=True,
            )

    def modify_decision(self,graph,decision,path):
        if not path:
            return int(decision)

        score=sum(
            self.edge_credit.get(
                edge,
                0.0,
            )
            for edge in path
        )

        if score<=-0.5:
            return 1-int(decision)

        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        delta=(
            1.0
            if predicted==answer
            else -1.0
        )

        for edge in path:
            self.edge_credit[edge]=max(
                -1.0,
                min(
                    1.0,
                    0.80*self.edge_credit.get(
                        edge,
                        0.0,
                    )
                    +0.20*delta,
                ),
            )

        self.count+=1


class TDError(CreditBase):
    """
    TD-like temporal-difference error.

    Improvement hypothesis:
    learn a value estimate and update only by prediction error instead of raw
    reward, giving credit a notion of surprise.
    """
    name="td_error"

    def __init__(self):
        self.value=0.0
        self.signal=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "td_value",
            "td_value",
            value=self.value,
            persistent=True,
        )
        graph.add_node(
            "credit_state",
            "credit_state",
            value=self.signal,
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        if self.signal>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        reward=(
            1.0
            if predicted==answer
            else -1.0
        )

        td_error=(
            reward
            -self.value
        )

        self.value=(
            self.value
            +0.30*td_error
        )

        if td_error>0:
            self.signal=(
                0.90*self.signal
                +0.10
            )
        else:
            self.signal=(
                0.90*self.signal
                +0.30
            )

        self.count+=1


class ReplayCredit(CreditBase):
    """
    Experience replay.

    Improvement hypothesis:
    retain a short queue of erroneous/relevant decisions and replay their
    credit signal before subsequent decisions.
    """
    name="replay"

    def __init__(self):
        self.replay:List[int]=[]
        self.signal=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "replay_state",
            "replay_state",
            value=self.signal,
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        if self.signal>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        error=int(
            predicted!=answer
        )

        self.replay.append(error)

        if len(self.replay)>4:
            self.replay.pop(0)

        self.signal=(
            sum(self.replay)
            /len(self.replay)
        )

        self.count+=1


class ContextualCredit(CreditBase):
    """
    Context-conditioned credit.

    Improvement hypothesis:
    global credit is too coarse; maintain separate policy error estimates for
    the observable context cue.
    """
    name="contextual"

    def __init__(self):
        self.by_context={
            0:0.0,
            1:0.0,
        }
        self.current_context=0
        self.count=0

    def inject_state(self,graph):
        cue=next(
            (
                n for n in graph.nodes.values()
                if n.role=="cue"
            ),
            None,
        )
        if cue is not None:
            self.current_context=int(
                cue.value>=0.5
            )

        graph.add_node(
            "context_credit",
            "context_credit",
            value=self.by_context[
                self.current_context
            ],
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        if self.by_context[
            self.current_context
        ]>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        context=int(
            episode.cue_bit>=0.5
        )

        error=float(
            predicted!=answer
        )

        self.by_context[context]=(
            0.65*
            self.by_context[context]
            +0.35*error
        )

        self.count+=1


class AdvantageCredit(CreditBase):
    """
    Baseline/advantage-style credit.

    Improvement hypothesis:
    subtract a running baseline so only unusually bad outcomes create a strong
    learning signal, reducing credit thrashing.
    """
    name="advantage"

    def __init__(self):
        self.baseline=0.0
        self.signal=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "advantage_state",
            "advantage_state",
            value=self.signal,
            persistent=True,
        )

    def modify_decision(self,graph,decision,path):
        if self.signal>=0.5:
            return 1-int(decision)
        return int(decision)

    def feedback(self,graph,predicted,answer,path,episode):
        reward=(
            1.0
            if predicted==answer
            else -1.0
        )

        advantage=(
            reward
            -self.baseline
        )

        self.baseline=(
            0.85*self.baseline
            +0.15*reward
        )

        if advantage<0:
            self.signal=min(
                1.0,
                self.signal
                +0.25*abs(advantage),
            )
        else:
            self.signal*=0.95

        self.count+=1


CREDITS={
    "none":NoneCredit,
    "global_reward":GlobalReward,
    "eligibility":Eligibility,
    "local_path":LocalPathCredit,
    "td_error":TDError,
    "replay":ReplayCredit,
    "contextual":ContextualCredit,
    "advantage":AdvantageCredit,
}
