
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import random


ACTIONS=("NO","YES")
TASKS=("credit",)


@dataclass
class Node:
    name:str
    role:str
    value:float=0.0
    persistent:bool=True


@dataclass
class Edge:
    source:str
    relation:str
    target:str
    weight:float=1.0
    persistent:bool=True


@dataclass
class Graph:
    nodes:Dict[str,Node]
    edges:List[Edge]

    def clone(self)->"Graph":
        return Graph(
            nodes={
                k:Node(
                    n.name,
                    n.role,
                    n.value,
                    n.persistent,
                )
                for k,n in self.nodes.items()
            },
            edges=[
                Edge(
                    e.source,
                    e.relation,
                    e.target,
                    e.weight,
                    e.persistent,
                )
                for e in self.edges
            ],
        )

    def add_node(
        self,
        name:str,
        role:str,
        value:float=0.0,
        persistent:bool=True,
    ):
        self.nodes[name]=Node(
            name,
            role,
            value,
            persistent,
        )

    def add_edge(
        self,
        source:str,
        relation:str,
        target:str,
        weight:float=1.0,
        persistent:bool=True,
    ):
        if source not in self.nodes:
            self.add_node(source,"opaque")
        if target not in self.nodes:
            self.add_node(target,"opaque")

        for e in self.edges:
            if (
                e.source==source
                and e.relation==relation
                and e.target==target
            ):
                e.weight=weight
                e.persistent=(
                    e.persistent or persistent
                )
                return

        self.edges.append(
            Edge(
                source,
                relation,
                target,
                weight,
                persistent,
            )
        )

    def outgoing(
        self,
        source:str,
        relation:str|None=None,
    )->List[Edge]:
        return [
            e
            for e in self.edges
            if e.source==source
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def decay(
        self,
        factor:float,
    ):
        for n in self.nodes.values():
            n.value*=factor
        for e in self.edges:
            if not e.persistent:
                e.weight*=factor


@dataclass(frozen=True)
class Query:
    source:str
    relation:str
    target:str


@dataclass(frozen=True)
class Episode:
    seed:int
    index:int
    graph:Graph
    query:Query
    observation:int
    cue:int
    latent_rule:int
    answer:int

    @property
    def action(self):
        return ACTIONS[self.answer]


@dataclass(frozen=True)
class Sequence:
    seed:int
    latent_rule:int
    episodes:Tuple[Episode,...]


def opaque(
    prefix:str,
    seed:int,
    index:int,
)->str:
    return (
        prefix+"_"
        +hashlib.sha256(
            f"{prefix}:{seed}:{index}".encode()
        ).hexdigest()[:10]
    )


def make_sequence(
    seed:int,
    episodes:int=12,
)->Sequence:
    """
    The environment chooses a hidden policy rule per sequence.

    latent_rule is deliberately NOT present in Query or the visible persistent
    graph. The only way to infer it is from delayed outcome feedback.
    """

    rng=random.Random(
        70001*seed
    )
    latent_rule=rng.randrange(2)

    rows=[]

    for index in range(episodes):
        graph=Graph(
            nodes={},
            edges=[],
        )

        source=opaque(
            "source",
            seed,
            index,
        )
        target=opaque(
            "target",
            seed,
            index,
        )

        graph.add_node(
            source,
            "source",
        )
        graph.add_node(
            target,
            "target",
        )

        # Each episode has an opaque observational feature. The correct action
        # depends on observation XOR latent_rule. The graph itself provides
        # no direct indication of the latent rule.
        observation=(
            seed*3
            +index*5
        )%2

        cue=(
            seed//3
            +index
        )%2

        obs_node=opaque(
            "observation",
            seed,
            index,
        )
        cue_node=opaque(
            "cue",
            seed,
            index,
        )

        graph.add_node(
            obs_node,
            "observation",
            value=float(observation),
            persistent=False,
        )
        graph.add_node(
            cue_node,
            "cue",
            value=float(cue),
            persistent=True,
        )

        graph.add_edge(
            cue_node,
            "modulates",
            target,
        )

        query=Query(
            source,
            rng.choice(
                ("r0","r1","r2")
            ),
            target,
        )

        # Distractor structure is independent of latent rule.
        distractor=opaque(
            "d",
            seed,
            index,
        )
        graph.add_node(
            distractor,
            "distractor",
        )
        graph.add_edge(
            source,
            "r3",
            distractor,
        )
        graph.add_edge(
            distractor,
            query.relation,
            target,
        )

        # Credit task: the frozen core policy is observation XOR cue.
        # The hidden sequence rule is an additional inversion. Therefore a
        # terminal error directly identifies whether the latent rule is active.
        answer=observation ^ cue ^ latent_rule

        rows.append(
            Episode(
                seed=seed,
                index=index,
                graph=graph,
                query=query,
                observation=observation,
                cue=cue,
                latent_rule=latent_rule,
                answer=answer,
            )
        )

    return Sequence(
        seed=seed,
        latent_rule=latent_rule,
        episodes=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Core state modules. Kept constant so this experiment focuses on credit.
# ---------------------------------------------------------------------------

class Memory:
    name="persistent"

    def write(
        self,
        graph:Graph,
    ):
        observation=next(
            (
                n
                for n in graph.nodes.values()
                if n.role=="observation"
            ),
            None,
        )

        graph.add_node(
            "memory",
            "memory",
            value=(
                1.0
                if observation is not None
                and observation.value>=0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(
        self,
        graph:Graph,
    ):
        node=graph.nodes.get("memory")
        if node is None:
            return

        if node.value>=0:
            node.value=max(
                node.value,
                0.85,
            )
        else:
            node.value=min(
                node.value,
                -0.85,
            )


class Dynamics:
    name="transform"

    def step(
        self,
        graph:Graph,
        step:int,
    ):
        graph.decay(0.97)


class Readout:
    name="memory"

    def read(
        self,
        graph:Graph,
    )->int:
        node=graph.nodes.get("memory")
        if node is None:
            return 0
        return int(
            node.value>=0
        )


class Planner:
    name="binding"

    def plan(
        self,
        graph:Graph,
        recalled:int,
    )->int:
        value=int(recalled)

        cue=next(
            (
                n
                for n in graph.nodes.values()
                if n.role=="cue"
            ),
            None,
        )

        if cue is not None and cue.value>=0.5:
            value^=1

        return value


class CoreSystem:
    def __init__(self,credit):
        self.memory=Memory()
        self.dynamics=Dynamics()
        self.readout=Readout()
        self.planner=Planner()
        self.credit=credit

    def run(
        self,
        episode:Episode,
        learn:bool=True,
    )->dict:
        graph=episode.graph.clone()

        self.memory.write(graph)

        # Observation disappears before decision.
        initial=next(
            (
                name
                for name in list(graph.nodes)
                if graph.nodes[name].role=="observation"
            ),
            None,
        )
        if initial is not None:
            graph.nodes.pop(initial)

        for step in range(1,5):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
            )

        recalled=self.readout.read(
            graph,
        )

        # Credit is part of the decision path. A learned policy signal can
        # transform the recalled state before planning. Without this, credit
        # would be a decorative post-hoc metric rather than cognition.
        credit_signal=self.credit.signal()
        if credit_signal:
            recalled=1-int(recalled)

        decision=self.planner.plan(
            graph,
            recalled,
        )

        correct=(
            decision==episode.answer
        )

        if learn:
            self.credit.update(
                decision,
                episode.answer,
            )

        return {
            "decision":decision,
            "answer":episode.answer,
            "correct":correct,
        }
