
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalConfig:
    name:str
    query_bias:float
    exploration:float
    settle_steps:int
    settle_gain:float
    settle_leak:float
    ambiguity_threshold:float
    low_confidence_threshold:float
    hypothesis_protection:float
    branch_separation:float
    counterfactual_bias:float


class HierarchicalAdaptiveRepresentation:
    """
    Representation controller conditioned on the *current hypothesis*.

    Representation does not decide what rule is correct. It decides how much
    internal state to retain:

        collapse  -> decisive one-state representation
        dual      -> preserve actual + alternative

    A strong hypothesis protects the current representation from being
    overwritten by generic ambiguity.  Counterfactual tasks explicitly retain
    both branches.
    """

    def __init__(self,config):
        self.config=config
        self.selected=0
        self.actual=0.0
        self.alternate=0.0
        self.mode="collapse"
        self.confidence=0.0
        self.count=0
        self.dual_count=0
        self.collapse_count=0

    def inject_state(self,graph):
        graph.add_node(
            "representation_mode",
            "representation_mode",
            value=1.0 if self.mode=="dual" else 0.0,
            persistent=True,
        )
        graph.add_node(
            "representation_confidence",
            "representation_confidence",
            value=self.confidence,
            persistent=True,
        )

    def _memory(self,graph):
        n=graph.nodes.get("memory")
        return int(
            n is not None
            and n.value>=0
        )

    def _cues(self,graph):
        out=[]
        for role in ("cue1","cue2","cue3"):
            node=next(
                (
                    n for n in graph.nodes.values()
                    if n.role==role
                ),
                None,
            )
            out.append(
                int(
                    node is not None
                    and node.value>=0.5
                )
            )
        return out

    def _choose_query(self,graph):
        cues=self._cues(graph)

        scores=[
            self.config.query_bias
            +self.config.exploration*(i+1)
            +0.10*bit
            for i,bit in enumerate(cues)
        ]

        ranked=sorted(
            enumerate(scores),
            key=lambda x:x[1],
            reverse=True,
        )

        self.selected=ranked[0][0]

        margin=max(
            0.0,
            min(
                1.0,
                ranked[0][1]-ranked[1][1],
            ),
        )

        return self.selected,cues,margin

    def _settle(self,memory,cue):
        x=0.5*(memory+cue)

        for _ in range(
            self.config.settle_steps
        ):
            drive=(
                self.config.settle_gain
                *(float(memory)+float(cue))
                -self.config.settle_leak*x
            )
            target=1.0 if drive>=0.5 else 0.0
            x=0.65*x+0.35*target

        return x

    def run(
        self,
        graph,
        episode,
        hypothesis_state,
        hypothesis_confidence,
    ):
        selected,cues,query_margin=self._choose_query(graph)

        memory=self._memory(graph)
        cue=cues[selected]

        actual=self._settle(
            memory,
            cue,
        )

        alternate=self._settle(
            memory,
            1-cue,
        )

        if abs(actual-alternate)<self.config.branch_separation:
            if actual>=alternate:
                actual=min(
                    1.0,
                    actual+self.config.branch_separation,
                )
            else:
                alternate=min(
                    1.0,
                    alternate+self.config.branch_separation,
                )

        actual_conf=max(actual,1.0-actual)
        alternate_conf=max(alternate,1.0-alternate)

        branch_disagreement=abs(
            actual-alternate
        )

        ambiguity=(
            (1.0-query_margin)
            +branch_disagreement
        )/2.0

        explicit_counterfactual=(
            episode.task=="counterfactual"
        )

        # Hierarchy: current hypothesis controls how willing representation is
        # to stay expanded.
        low_confidence=(
            hypothesis_confidence
            <self.config.low_confidence_threshold
        )

        protected=(
            hypothesis_state!=0
            and hypothesis_confidence
            >=self.config.hypothesis_protection
        )

        if explicit_counterfactual:
            mode="dual"
        elif low_confidence and (
            ambiguity>=self.config.ambiguity_threshold
        ):
            mode="dual"
        elif protected:
            mode="collapse"
        elif ambiguity>=self.config.ambiguity_threshold:
            mode="dual"
        else:
            mode="collapse"

        if mode=="collapse":
            # Never reinterpret a protected hypothesis just because the
            # alternate branch looks numerically attractive.
            if protected:
                bound=actual
            elif actual_conf>=alternate_conf:
                bound=actual
            else:
                bound=alternate
            self.collapse_count+=1

        else:
            if explicit_counterfactual:
                bound=(
                    (1-self.config.counterfactual_bias)*actual
                    +self.config.counterfactual_bias*alternate
                )
            else:
                bound=max(actual,alternate)
            self.dual_count+=1

        self.actual=actual
        self.alternate=alternate
        self.mode=mode

        self.confidence=(
            0.70*self.confidence
            +0.30*max(
                actual_conf,
                alternate_conf,
            )
        )

        self.count+=1

        return int(bound>=0.5)


CONFIGS={
    "hier_balanced":HierarchicalConfig(
        "hier_balanced",
        query_bias=0.60,
        exploration=0.10,
        settle_steps=4,
        settle_gain=0.80,
        settle_leak=0.25,
        ambiguity_threshold=0.35,
        low_confidence_threshold=0.55,
        hypothesis_protection=0.70,
        branch_separation=0.15,
        counterfactual_bias=0.75,
    ),
    "hier_protective":HierarchicalConfig(
        "hier_protective",
        query_bias=0.60,
        exploration=0.08,
        settle_steps=5,
        settle_gain=0.80,
        settle_leak=0.25,
        ambiguity_threshold=0.30,
        low_confidence_threshold=0.60,
        hypothesis_protection=0.60,
        branch_separation=0.15,
        counterfactual_bias=0.80,
    ),
    "hier_adaptive":HierarchicalConfig(
        "hier_adaptive",
        query_bias=0.65,
        exploration=0.05,
        settle_steps=5,
        settle_gain=0.85,
        settle_leak=0.20,
        ambiguity_threshold=0.45,
        low_confidence_threshold=0.50,
        hypothesis_protection=0.75,
        branch_separation=0.10,
        counterfactual_bias=0.70,
    ),
    "hier_dual_ready":HierarchicalConfig(
        "hier_dual_ready",
        query_bias=0.60,
        exploration=0.10,
        settle_steps=6,
        settle_gain=0.80,
        settle_leak=0.25,
        ambiguity_threshold=0.25,
        low_confidence_threshold=0.65,
        hypothesis_protection=0.80,
        branch_separation=0.20,
        counterfactual_bias=0.85,
    ),
}

assert len(CONFIGS)==4
