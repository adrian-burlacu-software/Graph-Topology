
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
            Node(source,"opaque"),
        )
        self.nodes.setdefault(
            target,
            Node(target,"opaque"),
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
    task:str
    graph:Graph
    query:Query
    answer_bit:int
    decision_step:int
    initial_bit:int
    cue_bit:int
    delta_bit:int
    operation_bit:int
    latent_rule:int

    @property
    def answer_action(self)->str:
        return ACTIONS[self.answer_bit]


@dataclass(frozen=True)
class Sequence:
    seed:int
    task:str
    latent_rule:int
    episodes:Tuple[Episode,...]


def opaque(prefix:str,seed:int,index:int)->str:
    return (
        prefix+"_"
        +hashlib.sha256(
            f"{prefix}:{seed}:{index}".encode()
        ).hexdigest()[:10]
    )


def make_sequence(
    seed:int,
    task:str,
    episodes:int=12,
    horizon:int=7,
)->Sequence:
    if task not in TASKS:
        raise ValueError(task)

    rng=random.Random(
        70001*seed
        +7919*TASKS.index(task)
    )

    latent_rule=rng.randrange(2)

    rows=[]

    for i in range(episodes):
        graph=Graph()

        source=opaque("q",seed,i*5)
        middle=opaque("q",seed,i*5+1)
        target=opaque("q",seed,i*5+2)
        decoy=opaque("d",seed,i*5+3)

        graph.add_node(
            source,
            "query_source",
        )
        graph.add_node(
            middle,
            "concept",
        )
        graph.add_node(
            target,
            "query_target",
        )
        graph.add_node(
            decoy,
            "concept",
        )

        rel_a,rel_b=rng.sample(
            ("r0","r1","r2","r3"),
            2,
        )
        decoy_rel=next(
            r for r in ("r0","r1","r2","r3")
            if r not in (rel_a,rel_b)
        )

        initial=(
            seed
            +i
            +latent_rule
        )%2
        cue=(
            seed//2
            +2*i
        )%2
        delta=(
            seed//3
            +i
        )%2
        operation=(
            seed//5
            +i
        )%2

        # Transient initial fact: memory has to encode it.
        fact=opaque(
            "fact",
            seed,
            i,
        )
        graph.add_node(
            fact,
            "initial_fact",
            value=float(initial),
            persistent=False,
        )

        cue_node=opaque(
            "cue",
            seed,
            i,
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

        delta_node=opaque(
            "delta",
            seed,
            i,
        )
        graph.add_node(
            delta_node,
            "delta",
            value=float(delta),
            persistent=True,
        )
        graph.add_edge(
            delta_node,
            "updates",
            target,
        )

        operation_node=opaque(
            "op",
            seed,
            i,
        )
        graph.add_node(
            operation_node,
            "operation",
            value=float(operation),
            persistent=True,
        )
        graph.add_edge(
            operation_node,
            "operation",
            target,
        )

        # Stable relational environment, identical with respect to initial bit.
        graph.add_edge(
            source,
            rel_a,
            middle,
        )
        graph.add_edge(
            middle,
            rel_b,
            target,
        )

        # Distractor route.
        graph.add_edge(
            source,
            decoy_rel,
            decoy,
        )
        graph.add_edge(
            decoy,
            rel_b,
            target,
        )

        if task=="binding" and operation:
            control=opaque("control",seed,i)
            marker=opaque("marker",seed,i)

            graph.add_node(
                control,
                "control",
            )
            graph.add_node(
                marker,
                "negate",
            )
            graph.add_edge(
                control,
                "mode",
                marker,
            )
            graph.add_edge(
                control,
                "applies",
                target,
            )

        if task=="planning":
            p1=opaque(
                "plan",
                seed,
                i*2,
            )
            p2=opaque(
                "plan",
                seed,
                i*2+1,
            )

            graph.add_node(
                p1,
                "plan_step",
            )
            graph.add_node(
                p2,
                "plan_step",
            )
            graph.add_edge(
                p1,
                "order",
                p2,
            )
            graph.add_edge(
                p1,
                "applies",
                target,
            )
            graph.add_edge(
                p2,
                "applies",
                target,
            )

        # Nuisance graph.
        nuisance=opaque(
            "noise",
            seed,
            i,
        )
        graph.add_node(
            nuisance,
            "noise",
        )

        for j in range(4):
            n=opaque(
                "n",
                seed+17*i,
                j,
            )
            graph.add_node(
                n,
                "noise",
            )
            graph.add_edge(
                nuisance,
                rng.choice(
                    ("r0","r1","r2","r3")
                ),
                n,
            )

        if task=="memory":
            answer=initial
        elif task=="binding":
            answer=initial^cue
            if operation:
                answer=1-answer
        elif task=="dynamics":
            answer=initial^delta
        elif task=="credit":
            answer=initial^cue^latent_rule
        elif task=="planning":
            answer=initial^cue^operation
        else:
            raise AssertionError(task)

        rows.append(
            Episode(
                seed=seed,
                task=task,
                graph=graph,
                query=Query(
                    source,
                    rel_a,
                    rel_b,
                    target,
                ),
                answer_bit=answer,
                decision_step=horizon-1,
                initial_bit=initial,
                cue_bit=cue,
                delta_bit=delta,
                operation_bit=operation,
                latent_rule=latent_rule,
            )
        )

    return Sequence(
        seed=seed,
        task=task,
        latent_rule=latent_rule,
        episodes=tuple(rows),
    )


class PersistentMemory:
    def observe(self,graph:Graph,query:Query):
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

    def maintain(self,graph:Graph):
        node=graph.nodes.get("memory")
        if node is None:
            return

        if node.value>=0:
            node.value=max(
                0.85,
                node.value,
            )
        else:
            node.value=min(
                -0.85,
                node.value,
            )


class TransformDynamics:
    def step(
        self,
        graph:Graph,
        step:int,
    ):
        graph.decay(0.97)

        delta=next(
            (
                n for n in graph.nodes.values()
                if n.role=="delta"
            ),
            None,
        )

        if delta is None or delta.value<0.5:
            return

        node=graph.nodes.get("memory")
        if node is not None and step>=2:
            node.value*=-1.0


class MemoryReadout:
    def read(self,graph:Graph)->Tuple[int,List[Tuple]]:
        node=graph.nodes.get("memory")
        if node is None:
            return 0,[]
        return (
            int(node.value>=0),
            [],
        )


class BindingPlanner:
    def plan(
        self,
        graph:Graph,
        recalled:int,
    )->int:
        value=int(recalled)

        for node in graph.nodes.values():
            if node.role=="cue":
                if node.value>=0.5:
                    value^=1

            if node.role=="operation":
                if node.value>=0.5:
                    value^=1

        return value


class CoreSystem:
    """
    Frozen core from V296:
        persistent memory
        transform dynamics
        memory readout
        binding planner

    V297 only changes credit.
    """

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

        # Persistent credit state is intentionally inserted into the graph,
        # where later cognition can actually consume it.
        self.credit.inject_state(
            graph
        )

        self.memory.observe(
            graph,
            episode.query,
        )

        transient=next(
            (
                name for name in list(graph.nodes)
                if graph.nodes[name].role=="initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(transient)

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
