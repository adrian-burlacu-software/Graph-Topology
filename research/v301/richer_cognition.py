
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib
import random


ACTIONS=("NO","YES")

TASKS=(
    "delayed_memory",
    "sequence_binding",
    "interference",
    "rule_change",
    "planning",
    "counterfactual",
)


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

    def clone(self):
        return Graph(
            nodes={
                k:Node(
                    n.name,n.role,n.value,n.persistent
                )
                for k,n in self.nodes.items()
            },
            edges=[
                Edge(
                    e.source,e.relation,e.target,
                    e.weight,e.persistent
                )
                for e in self.edges
            ],
        )

    def add_node(
        self,
        name,
        role,
        value=0.0,
        persistent=True,
    ):
        self.nodes[name]=Node(
            name,
            role,
            value,
            persistent,
        )

    def add_edge(
        self,
        source,
        relation,
        target,
        weight=1.0,
        persistent=True,
    ):
        self.nodes.setdefault(
            source,
            Node(source,"opaque"),
        )
        self.nodes.setdefault(
            target,
            Node(target,"opaque"),
        )
        self.edges.append(
            Edge(
                source,
                relation,
                target,
                weight,
                persistent,
            )
        )

    def decay(self,factor):
        for node in self.nodes.values():
            node.value*=factor

    def incoming(
        self,
        target,
        relation=None,
    ):
        return [
            e for e in self.edges
            if e.target==target
            and e.weight>0
            and (
                relation is None
                or e.relation==relation
            )
        ]

    def outgoing(
        self,
        source,
        relation=None,
    ):
        return [
            e for e in self.edges
            if e.source==source
            and e.weight>0
            and (
                relation is None
                or e.relation==relation
            )
        ]


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
    task:str
    graph:Graph
    query:Query
    answer_bit:int
    decision_step:int
    initial_bit:int
    cue_bits:Tuple[int,int,int]
    latent_rule:int
    rule_version:int


@dataclass(frozen=True)
class Sequence:
    seed:int
    episodes:Tuple[Episode,...]


def opaque(prefix,seed,index):
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
    horizon:int=9,
):
    if task not in TASKS:
        raise ValueError(task)

    rng=random.Random(
        91009*seed
        +7919*TASKS.index(task)
    )

    latent_rule=rng.randrange(2)

    # Rule-change task contains two phases. The second phase requires the
    # learner to detect that the previously learned rule is no longer valid.
    change_at=max(
        2,
        episodes//2,
    )

    rows=[]

    for index in range(episodes):
        graph=Graph()

        source=opaque("source",seed,index)
        middle=opaque("middle",seed,index)
        target=opaque("target",seed,index)
        decoy=opaque("decoy",seed,index)

        for name,role in (
            (source,"query_source"),
            (middle,"concept"),
            (target,"query_target"),
            (decoy,"concept"),
        ):
            graph.add_node(name,role)

        relation_a,relation_b=rng.sample(
            ("r0","r1","r2","r3"),
            2,
        )
        decoy_relation=next(
            r for r in ("r0","r1","r2","r3")
            if r not in (
                relation_a,
                relation_b,
            )
        )

        initial=(
            seed
            +index*3
            +latent_rule
        )%2

        cue1=(
            seed//2
            +index
        )%2

        cue2=(
            seed//3
            +2*index
        )%2

        cue3=(
            seed//5
            +index
        )%2

        fact=opaque("fact",seed,index)
        graph.add_node(
            fact,
            "initial_fact",
            value=float(initial),
            persistent=False,
        )

        for slot,value,relation in (
            ("cue1",cue1,"cue"),
            ("cue2",cue2,"cue"),
            ("cue3",cue3,"cue"),
        ):
            node=opaque(
                slot,
                seed,
                index,
            )
            graph.add_node(
                node,
                slot,
                value=float(value),
                persistent=True,
            )
            graph.add_edge(
                node,
                relation,
                target,
            )

        # Query topology is independent of the answer-bearing bits.
        graph.add_edge(
            source,
            relation_a,
            middle,
        )
        graph.add_edge(
            middle,
            relation_b,
            target,
        )

        # Two distractor routes create interference.
        graph.add_edge(
            source,
            decoy_relation,
            decoy,
        )
        graph.add_edge(
            decoy,
            relation_a,
            target,
        )

        decoy2=opaque("decoy2",seed,index)
        graph.add_node(
            decoy2,
            "concept",
        )
        other_relation=next(
            r for r in ("r0","r1","r2","r3")
            if r not in (
                relation_a,
                relation_b,
                decoy_relation,
            )
        )
        graph.add_edge(
            source,
            other_relation,
            decoy2,
        )
        graph.add_edge(
            decoy2,
            relation_b,
            target,
        )

        # Ordered plan structure.
        for step_no in range(3):
            plan=opaque(
                "plan",
                seed,
                index*3+step_no,
            )
            graph.add_node(
                plan,
                "plan_step",
            )
            graph.add_edge(
                plan,
                "applies",
                target,
            )

        # Rule-change marker appears only when phase changes.
        rule_version=(
            1
            if task=="rule_change"
            and index>=change_at
            else 0
        )

        if rule_version:
            marker=opaque(
                "rule_change",
                seed,
                index,
            )
            graph.add_node(
                marker,
                "rule_change_marker",
                value=1.0,
            )

        # Counterfactual graph control.
        if task=="counterfactual":
            control=opaque(
                "control",
                seed,
                index,
            )
            negate=opaque(
                "negate",
                seed,
                index,
            )
            graph.add_node(
                control,
                "control",
            )
            graph.add_node(
                negate,
                "negate",
            )
            graph.add_edge(
                control,
                "mode",
                negate,
            )
            graph.add_edge(
                control,
                "applies",
                target,
            )

        # Larger nuisance subgraph.
        for j in range(8):
            noise=opaque(
                "noise",
                seed+31*index,
                j,
            )
            graph.add_node(
                noise,
                "noise",
            )
            graph.add_edge(
                decoy,
                rng.choice(
                    ("r0","r1","r2","r3")
                ),
                noise,
            )

        # Answer functions intentionally differ by cognitive task.
        if task=="delayed_memory":
            answer=initial

        elif task=="sequence_binding":
            answer=(
                initial
                ^cue1
                ^cue2
                ^cue3
            )

        elif task=="interference":
            answer=(
                initial
                ^cue1
            )

        elif task=="rule_change":
            active_rule=(
                latent_rule
                if rule_version==0
                else 1-latent_rule
            )
            answer=initial^active_rule

        elif task=="planning":
            answer=(
                initial
                ^cue1
                ^cue2
                ^cue3
            )

        elif task=="counterfactual":
            answer=(
                initial^cue1
            )
            if cue2:
                answer=1-answer

        else:
            raise AssertionError(task)

        rows.append(
            Episode(
                seed=seed,
                index=index,
                task=task,
                graph=graph,
                query=Query(
                    source,
                    relation_a,
                    relation_b,
                    target,
                ),
                answer_bit=answer,
                decision_step=horizon-1,
                initial_bit=initial,
                cue_bits=(
                    cue1,
                    cue2,
                    cue3,
                ),
                latent_rule=latent_rule,
                rule_version=rule_version,
            )
        )

    return Sequence(
        seed=seed,
        episodes=tuple(rows),
    )


# ---------------------------------------------------------------------------
# Frozen architecture under test
# ---------------------------------------------------------------------------

class PersistentMemory:
    def observe(self,graph,query):
        fact=next(
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
                if fact is not None
                and fact.value>=0.5
                else -1.0
            ),
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

        # Three later operations create genuine temporal composition.
        deltas=[
            n for n in graph.nodes.values()
            if n.role=="cue2"
        ]

        if deltas and deltas[0].value>=0.5:
            memory=graph.nodes.get("memory")
            if memory is not None:
                memory.value*=-1.0


class MemoryReadout:
    def read(self,graph):
        node=graph.nodes.get("memory")
        if node is None:
            return 0,[]
        return int(node.value>=0),[]


class BindingPlanner:
    def plan(self,graph,recalled):
        value=int(recalled)

        for role in (
            "cue1",
            "cue2",
            "cue3",
        ):
            for node in graph.nodes.values():
                if (
                    node.role==role
                    and node.value>=0.5
                ):
                    value^=1

        return value


class RichCognitiveSystem:
    def __init__(self,credit):
        self.memory=PersistentMemory()
        self.dynamics=TransformDynamics()
        self.readout=MemoryReadout()
        self.planner=BindingPlanner()
        self.credit=credit

    def run(self,episode,learn=True):
        graph=episode.graph.clone()

        self.credit.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient=next(
            (
                name for name in list(graph.nodes)
                if graph.nodes[name].role
                    =="initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(
                transient
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
        }
