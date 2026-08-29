
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import random


ACTIONS=("NO","YES")
TASKS=("memory","binding","dynamics","credit","planning")


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
    nodes:Dict[str,Node]=field(default_factory=dict)
    edges:List[Edge]=field(default_factory=list)

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
        self.nodes.setdefault(
            source,
            Node(
                source,
                "opaque",
            ),
        )
        self.nodes.setdefault(
            target,
            Node(
                target,
                "opaque",
            ),
        )

        for edge in self.edges:
            if (
                edge.source==source
                and edge.relation==relation
                and edge.target==target
            ):
                edge.weight=weight
                edge.persistent=(
                    edge.persistent or persistent
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
        relation:Optional[str]=None,
    )->List[Edge]:
        return [
            e for e in self.edges
            if e.source==source
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def incoming(
        self,
        target:str,
        relation:Optional[str]=None,
    )->List[Edge]:
        return [
            e for e in self.edges
            if e.target==target
            and e.weight>0.0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def decay(self,factor:float):
        for node in self.nodes.values():
            node.value*=factor

        for edge in self.edges:
            if not edge.persistent:
                edge.weight*=factor


@dataclass(frozen=True)
class Query:
    source:str
    relation_a:str
    relation_b:str
    target:str


@dataclass(frozen=True)
class Episode:
    seed:int
    index:int
    graph:Graph
    query:Query
    answer_bit:int
    decision_step:int
    initial_bit:int
    cue_bit:int
    latent_rule:int

    @property
    def answer_action(self)->str:
        return ACTIONS[self.answer_bit]


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
    horizon:int=7,
)->Sequence:
    rng=random.Random(
        70001*seed
    )

    latent_rule=rng.randrange(2)
    rows=[]

    for index in range(episodes):
        graph=Graph()

        source=opaque("source",seed,index)
        target=opaque("target",seed,index)

        graph.add_node(
            source,
            "source",
        )
        graph.add_node(
            target,
            "target",
        )

        observation=(
            seed*3+index*5
        )%2

        cue=(
            seed//3+index
        )%2

        fact=opaque(
            "fact",
            seed,
            index,
        )
        cue_node=opaque(
            "cue",
            seed,
            index,
        )

        graph.add_node(
            fact,
            "initial_fact",
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

        relation=rng.choice(
            ("r0","r1","r2")
        )

        query=Query(
            source,
            relation,
            target,
            target,
        )

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
            relation,
            target,
        )

        # Credit-specific task: current episode alone is insufficient.
        answer=(
            observation
            ^ cue
            ^ latent_rule
        )

        rows.append(
            Episode(
                seed=seed,
                index=index,
                graph=graph,
                query=query,
                answer_bit=answer,
                decision_step=horizon-1,
                initial_bit=observation,
                cue_bit=cue,
                latent_rule=latent_rule,
            )
        )

    return Sequence(
        seed=seed,
        latent_rule=latent_rule,
        episodes=tuple(rows),
    )


class PersistentMemory:
    def observe(self,graph,query):
        sensory=next(
            (
                n for n in graph.nodes.values()
                if n.role=="initial_fact"
            ),
            None,
        )

        graph.add_node(
            "memory",
            "memory",
            value=(
                1.0
                if sensory is not None
                and sensory.value>=0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self,graph):
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


class TransformDynamics:
    def step(self,graph,step):
        graph.decay(0.97)


class MemoryReadout:
    def read(self,graph):
        node=graph.nodes.get("memory")
        if node is None:
            return 0,[]
        return (
            int(node.value>=0),
            [],
        )


class BindingPlanner:
    def plan(self,graph,recalled):
        value=int(recalled)

        cue=next(
            (
                n for n in graph.nodes.values()
                if n.role=="cue"
            ),
            None,
        )

        if cue is not None and cue.value>=0.5:
            value^=1

        return value


class CoreSystem:
    def __init__(self,credit):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.readout=MemoryReadout()
        self.planner=BindingPlanner()
        self.credit=credit

    def run(
        self,
        episode:Episode,
        learn:bool=True,
    )->dict:
        graph=episode.graph.clone()

        self.credit.inject_state(
            graph
        )

        self.memory.observe(
            graph,
            episode.query,
        )

        initial=next(
            (
                name
                for name in list(graph.nodes)
                if graph.nodes[name].role
                    =="initial_fact"
            ),
            None,
        )

        if initial is not None:
            graph.nodes.pop(
                initial
            )

        for step in range(
            1,
            episode.decision_step+1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(
                graph,
            )

        recalled,path=self.readout.read(
            graph,
        )

        decision=self.planner.plan(
            graph,
            recalled,
        )

        decision=self.credit.modify_decision(
            graph,
            decision,
            path,
        )

        correct=(
            decision==episode.answer_bit
        )

        if learn:
            self.credit.feedback(
                graph,
                decision,
                episode.answer_bit,
                path,
                episode,
            )

        return {
            "correct":correct,
            "decision":decision,
            "answer":episode.answer_bit,
            "path":path,
        }
