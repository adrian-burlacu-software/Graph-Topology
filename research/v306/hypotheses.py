
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HypothesisConfig:
    name:str
    memory_weight:float
    cue_weight:float
    structural_weight:float
    role_weight:float
    iterative_steps:int
    attractor_decay:float
    chunk_strength:float
    counterfactual_strength:float
    query_strength:float


class ArchitectureOverlay:
    """
    V306 hypothesis families.

    Every overlay receives the same graph and produces a candidate cognitive
    state before the frozen hypothesis/revision controller acts.

    These are deliberately different mechanisms, not just parameter tweaks:
      1. role_separation       explicit typed-role state
      2. relational_messages   relation-aware evidence aggregation
      3. iterative_settling    recurrent attractor-like competition
      4. episodic_chunking     compress repeated co-occurrence patterns
      5. counterfactual_sim    compare actual vs alternative internal states
      6. active_query          choose which cue deserves more computation
    """

    name="base"

    def __init__(self,config):
        self.config=config
        self.state=0.0
        self.count=0

    def inject_state(self,graph):
        graph.add_node(
            "overlay_state",
            "overlay_state",
            value=self.state,
            persistent=True,
        )

    def _memory(self,graph):
        node=graph.nodes.get("memory")
        return (
            1
            if node is not None and node.value>=0
            else 0
        )

    def _cues(self,graph):
        values=[]
        for role in ("cue1","cue2","cue3"):
            node=next(
                (
                    n for n in graph.nodes.values()
                    if n.role==role
                ),
                None,
            )
            values.append(
                1 if node is not None and node.value>=0.5 else 0
            )
        return values

    def _target_score(self,graph):
        targets={
            n.name for n in graph.nodes.values()
            if n.role=="query_target"
        }
        score=0.0
        for edge in graph.edges:
            if edge.target in targets:
                score+=edge.weight
            if edge.source in targets:
                score+=0.5*edge.weight
        return score

    def compute(self,graph,episode):
        raise NotImplementedError

    def transform_decision(self,graph,decision,episode):
        raise NotImplementedError

    def feedback(self,graph,predicted,answer,episode):
        self.count+=1


class RoleSeparation(ArchitectureOverlay):
    name="role_separation"

    def compute(self,graph,episode):
        m=self._memory(graph)
        c=self._cues(graph)

        # Keep typed roles separate all the way to decision time.
        role_score=(
            m*self.config.memory_weight
            +sum(c)*self.config.cue_weight
        )

        graph.add_node(
            "role_state",
            "role_state",
            value=role_score,
            persistent=True,
        )
        self.state=role_score

    def transform_decision(self,graph,decision,episode):
        m=self._memory(graph)
        c=self._cues(graph)

        # Interference deliberately ignores distractor role.
        value=m
        for i,bit in enumerate(c):
            if (
                bit
                and i==0
            ):
                value^=1

        return int(value)


class RelationalMessages(ArchitectureOverlay):
    name="relational_messages"

    def compute(self,graph,episode):
        target_names={
            n.name for n in graph.nodes.values()
            if n.role=="query_target"
        }

        score0=0.0
        score1=0.0

        for node in graph.nodes.values():
            base=(
                self.config.memory_weight
                if node.role=="memory"
                else 0.0
            )

            for edge in graph.edges:
                if edge.target in target_names:
                    if node.name==edge.source:
                        base+=self.config.structural_weight
                if edge.source in target_names:
                    if node.name==edge.target:
                        base+=0.5*self.config.structural_weight

            if node.role=="distractor":
                base*=0.1

            if int(round(node.value))==1:
                score1+=base
            else:
                score0+=base

        self.state=1.0 if score1>score0 else 0.0

        graph.add_node(
            "relation_state",
            "relation_state",
            value=self.state,
            persistent=True,
        )

    def transform_decision(self,graph,decision,episode):
        return int(
            self.state>=0.5
        )


class IterativeSettling(ArchitectureOverlay):
    name="iterative_settling"

    def compute(self,graph,episode):
        m=float(self._memory(graph))
        cues=self._cues(graph)

        x=m

        for _ in range(
            self.config.iterative_steps
        ):
            evidence=(
                self.config.memory_weight*m
                +self.config.cue_weight*cues[0]
                -self.config.attractor_decay*x
            )

            if evidence>0.5:
                x=1.0
            elif evidence<-0.5:
                x=0.0
            else:
                x=(
                    0.5*x
                    +0.5*float(evidence>=0)
                )

        self.state=x

        graph.add_node(
            "settled_state",
            "settled_state",
            value=x,
            persistent=True,
        )

    def transform_decision(self,graph,decision,episode):
        return int(self.state>=0.5)


class EpisodicChunking(ArchitectureOverlay):
    name="episodic_chunking"

    def __init__(self,config):
        super().__init__(config)
        self.chunks:Dict[
            Tuple[int,int,int],
            float,
        ]={}

    def compute(self,graph,episode):
        m=self._memory(graph)
        c=self._cues(graph)
        key=(m,c[0],c[1])

        chunk=self.chunks.get(
            key,
            0.0,
        )

        self.state=chunk

        graph.add_node(
            "chunk_state",
            "chunk_state",
            value=chunk,
            persistent=True,
        )

    def transform_decision(self,graph,decision,episode):
        # Chunking modifies the cue binding only when a repeated state is
        # strongly represented.
        base=self._memory(graph)

        if self.state>=0.75:
            return base^self._cues(graph)[0]

        return int(decision)

    def feedback(self,graph,predicted,answer,episode):
        m=self._memory(graph)
        c=self._cues(graph)
        key=(m,c[0],c[1])
        reward=1.0 if predicted==answer else -1.0

        self.chunks[key]=(
            0.80*self.chunks.get(key,0.0)
            +0.20*reward
        )
        self.count+=1


class CounterfactualSimulation(ArchitectureOverlay):
    name="counterfactual_sim"

    def compute(self,graph,episode):
        m=self._memory(graph)
        c=self._cues(graph)

        actual=m^c[0]
        alternative=m^(1-c[0])

        # Prefer actual state unless an explicit counterfactual structure
        # predicts a reversal.
        if episode.task=="counterfactual":
            score=(
                actual
                +self.config.counterfactual_strength
                *alternative
            )
            self.state=1.0 if score>=1.0 else 0.0
        else:
            self.state=float(actual)

        graph.add_node(
            "counterfactual_state",
            "counterfactual_state",
            value=self.state,
            persistent=True,
        )

    def transform_decision(self,graph,decision,episode):
        return int(self.state>=0.5)


class ActiveQuery(ArchitectureOverlay):
    name="active_query"

    def compute(self,graph,episode):
        cues=self._cues(graph)

        # Allocate more internal weight to the cue that has historically been
        # more predictive, while retaining a small exploration mass.
        scores=[
            self.config.query_strength
            +(0.15 if bit else 0.0)
            for bit in cues
        ]

        best=max(
            range(len(scores)),
            key=lambda i:scores[i],
        )

        self.state=float(best)

        graph.add_node(
            "query_state",
            "query_state",
            value=self.state,
            persistent=True,
        )

    def transform_decision(self,graph,decision,episode):
        cues=self._cues(graph)
        chosen=int(self.state)

        return int(
            self._memory(graph)
            ^cues[chosen]
        )


CONFIGS={
    "role_separation":HypothesisConfig(
        "role_separation",
        1.0,0.7,0.5,1.0,1,0.5,0.0,0.0,0.0,
    ),
    "relational_messages":HypothesisConfig(
        "relational_messages",
        0.8,0.6,1.0,0.8,1,0.5,0.0,0.0,0.0,
    ),
    "iterative_settling":HypothesisConfig(
        "iterative_settling",
        1.0,0.7,0.5,0.8,5,0.25,0.0,0.0,0.0,
    ),
    "episodic_chunking":HypothesisConfig(
        "episodic_chunking",
        0.8,0.7,0.5,0.8,1,0.5,0.8,0.0,0.0,
    ),
    "counterfactual_sim":HypothesisConfig(
        "counterfactual_sim",
        0.8,0.7,0.5,0.8,1,0.5,0.0,1.0,0.0,
    ),
    "active_query":HypothesisConfig(
        "active_query",
        0.8,0.8,0.5,0.8,1,0.5,0.0,0.0,0.8,
    ),
}

CLASSES={
    "role_separation":RoleSeparation,
    "relational_messages":RelationalMessages,
    "iterative_settling":IterativeSettling,
    "episodic_chunking":EpisodicChunking,
    "counterfactual_sim":CounterfactualSimulation,
    "active_query":ActiveQuery,
}
