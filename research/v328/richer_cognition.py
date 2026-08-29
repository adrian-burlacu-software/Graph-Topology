
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import hashlib
import random


ACTIONS = ("NO", "YES")

TASKS = (
    "delayed_memory",
    "sequence_binding",
    "interference",
    "rule_change",
    "planning",
    "counterfactual",
)


@dataclass
class Node:
    name: str
    role: str
    value: float = 0.0
    persistent: bool = True


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    weight: float = 1.0
    persistent: bool = True


@dataclass
class Graph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def clone(self) -> "Graph":
        return Graph(
            nodes={
                k: Node(
                    n.name,
                    n.role,
                    n.value,
                    n.persistent,
                )
                for k, n in self.nodes.items()
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
        name: str,
        role: str,
        value: float = 0.0,
        persistent: bool = True,
    ):
        self.nodes[name] = Node(
            name,
            role,
            value,
            persistent,
        )

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        persistent: bool = True,
    ):
        self.nodes.setdefault(
            source,
            Node(source, "opaque"),
        )
        self.nodes.setdefault(
            target,
            Node(target, "opaque"),
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

    def incoming(
        self,
        target: str,
        relation: Optional[str] = None,
    ) -> List[Edge]:
        return [
            e
            for e in self.edges
            if e.target == target
            and e.weight > 0.0
            and (
                relation is None
                or e.relation == relation
            )
        ]

    def outgoing(
        self,
        source: str,
        relation: Optional[str] = None,
    ) -> List[Edge]:
        return [
            e
            for e in self.edges
            if e.source == source
            and e.weight > 0.0
            and (
                relation is None
                or e.relation == relation
            )
        ]

    def remove_node(self, name: str):
        self.nodes.pop(name, None)
        self.edges = [
            e for e in self.edges
            if e.source != name
            and e.target != name
        ]

    def decay(self, factor: float):
        for node in self.nodes.values():
            node.value *= factor


@dataclass(frozen=True)
class Query:
    source: str
    relation_a: str
    relation_b: str
    target: str


@dataclass(frozen=True)
class Episode:
    seed: int
    index: int
    task: str
    graph: Graph
    query: Query
    answer_bit: int
    decision_step: int
    initial_bit: int
    cue_bits: Tuple[int, int, int]
    distractor_bit: int
    latent_rule: int
    rule_version: int
    counterfactual_bit: int

    @property
    def answer_action(self) -> str:
        return ACTIONS[self.answer_bit]


@dataclass(frozen=True)
class Sequence:
    seed: int
    task: str
    episodes: Tuple[Episode, ...]
    initial_rule: int


def opaque(prefix: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(
        f"{prefix}:{seed}:{index}".encode()
    ).hexdigest()[:10]
    return f"{prefix}_{digest}"


def make_sequence(
    seed: int,
    task: str,
    episodes: int = 12,
    horizon: int = 9,
) -> Sequence:
    if task not in TASKS:
        raise ValueError(task)

    rng = random.Random(
        91009 * seed
        + 7919 * TASKS.index(task)
    )

    initial_rule = rng.randrange(2)
    change_at = max(3, episodes // 2)

    rows = []

    for index in range(episodes):
        graph = Graph()

        source = opaque("source", seed, index)
        middle = opaque("middle", seed, index)
        target = opaque("target", seed, index)

        graph.add_node(source, "query_source")
        graph.add_node(middle, "concept")
        graph.add_node(target, "query_target")

        relation_a, relation_b = rng.sample(
            ("r0", "r1", "r2", "r3"),
            2,
        )

        initial = (
            seed
            + 3 * index
            + initial_rule
        ) % 2

        cue1 = (
            seed // 2
            + index
        ) % 2

        cue2 = (
            seed // 3
            + 2 * index
        ) % 2

        cue3 = (
            seed // 5
            + index
        ) % 2

        # Rule-change isolates the rule itself. The ordinary binding cues are
        # zero so the frozen core is correct in phase 1 and predictably wrong
        # after the environment changes the rule.
        if task == "rule_change":
            cue1 = 0
            cue2 = 0
            cue3 = 0

        distractor = (
            seed
            + index * 7
        ) % 2

        counterfactual = (
            seed // 7
            + index
        ) % 2

        fact = opaque("fact", seed, index)
        graph.add_node(
            fact,
            "initial_fact",
            value=float(initial),
            persistent=False,
        )

        for role, value, relation in (
            ("cue1", cue1, "cue"),
            ("cue2", cue2, "cue"),
            ("cue3", cue3, "cue"),
        ):
            node = opaque(
                role,
                seed,
                index,
            )
            graph.add_node(
                node,
                role,
                value=float(value),
                persistent=True,
            )
            graph.add_edge(
                node,
                relation,
                target,
            )

        # A distractor shares the same target but has a different causal role.
        distractor_node = opaque(
            "distractor",
            seed,
            index,
        )
        graph.add_node(
            distractor_node,
            "distractor",
            value=float(distractor),
            persistent=True,
        )
        graph.add_edge(
            distractor_node,
            "distracts",
            target,
        )

        # Stable relational topology is deliberately independent of answer.
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

        # Decoy path.
        decoy = opaque(
            "decoy",
            seed,
            index,
        )
        graph.add_node(
            decoy,
            "concept",
        )
        graph.add_edge(
            source,
            rng.choice(("r0", "r1", "r2", "r3")),
            decoy,
        )
        graph.add_edge(
            decoy,
            relation_b,
            target,
        )

        # Rule revision cue. The marker itself doesn't say what the new rule is.
        rule_version = (
            1
            if task == "rule_change"
            and index >= change_at
            else 0
        )

        if rule_version:
            marker = opaque(
                "rule_change",
                seed,
                index,
            )
            graph.add_node(
                marker,
                "rule_change_marker",
                value=1.0,
                persistent=True,
            )

        # Counterfactual control graph.
        if task == "counterfactual":
            control = opaque(
                "control",
                seed,
                index,
            )
            negate = opaque(
                "negate",
                seed,
                index,
            )

            graph.add_node(
                control,
                "control",
                persistent=True,
            )
            graph.add_node(
                negate,
                "negate",
                persistent=True,
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

        # Larger nuisance graph.
        for j in range(10):
            noise = opaque(
                "noise",
                seed + 31 * index,
                j,
            )
            graph.add_node(
                noise,
                "noise",
            )
            graph.add_edge(
                distractor_node,
                rng.choice(
                    ("r0", "r1", "r2", "r3")
                ),
                noise,
            )

        active_rule = (
            1 - initial_rule
            if rule_version
            else initial_rule
        )

        if task == "delayed_memory":
            answer = initial

        elif task == "sequence_binding":
            answer = (
                initial
                ^ cue1
                ^ cue2
                ^ cue3
            )

        elif task == "interference":
            # The distractor is irrelevant by construction.
            answer = (
                initial
                ^ cue1
            )

        elif task == "rule_change":
            answer = initial ^ active_rule

        elif task == "planning":
            answer = (
                initial
                ^ cue1
                ^ cue2
                ^ cue3
            )

        elif task == "counterfactual":
            answer = initial ^ cue1
            if counterfactual:
                answer = 1 - answer

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
                decision_step=horizon - 1,
                initial_bit=initial,
                cue_bits=(cue1, cue2, cue3),
                distractor_bit=distractor,
                latent_rule=initial_rule,
                rule_version=rule_version,
                counterfactual_bit=counterfactual,
            )
        )

    return Sequence(
        seed=seed,
        task=task,
        episodes=tuple(rows),
        initial_rule=initial_rule,
    )


# ---------------------------------------------------------------------------
# Frozen successful core
# ---------------------------------------------------------------------------

class PersistentMemory:
    def observe(self, graph: Graph, query: Query):
        fact = next(
            (
                n for n in graph.nodes.values()
                if n.role == "initial_fact"
            ),
            None,
        )

        graph.add_node(
            "memory",
            "memory",
            value=(
                1.0
                if fact is not None
                and fact.value >= 0.5
                else -1.0
            ),
            persistent=True,
        )

    def maintain(self, graph: Graph):
        node = graph.nodes.get("memory")
        if node is None:
            return

        if node.value >= 0:
            node.value = max(node.value, 0.85)
        else:
            node.value = min(node.value, -0.85)


class TransformDynamics:
    def step(self, graph: Graph, step: int):
        graph.decay(0.97)


class MemoryReadout:
    def read(
        self,
        graph: Graph,
    ) -> Tuple[int, List[str]]:
        node = graph.nodes.get("memory")
        if node is None:
            return 0, []
        return int(node.value >= 0), ["memory"]


class BindingPlanner:
    def plan(
        self,
        graph: Graph,
        recalled: int,
    ) -> int:
        value = int(recalled)

        for role in (
            "cue1",
            "cue2",
            "cue3",
        ):
            for node in graph.nodes.values():
                if (
                    node.role == role
                    and node.value >= 0.5
                ):
                    value ^= 1

        return value


class FrozenCore:
    def __init__(self, cognitive_overlay):
        self.memory = PersistentMemory()
        self.dynamics = TransformDynamics()
        self.readout = MemoryReadout()
        self.planner = BindingPlanner()
        self.overlay = cognitive_overlay

    def run(
        self,
        episode: Episode,
        learn: bool = True,
    ) -> dict:
        graph = episode.graph.clone()

        self.overlay.inject_state(graph)

        self.memory.observe(
            graph,
            episode.query,
        )

        transient = next(
            (
                name
                for name in list(graph.nodes)
                if graph.nodes[name].role == "initial_fact"
            ),
            None,
        )

        if transient is not None:
            graph.nodes.pop(transient)

        for step in range(
            1,
            episode.decision_step + 1,
        ):
            self.dynamics.step(
                graph,
                step,
            )
            self.memory.maintain(graph)

        recalled, path = self.readout.read(
            graph,
        )

        decision = self.planner.plan(
            graph,
            recalled,
        )

        decision = self.overlay.transform_decision(
            graph,
            decision,
            episode,
        )

        correct = (
            decision == episode.answer_bit
        )

        if learn:
            self.overlay.feedback(
                graph,
                decision,
                episode.answer_bit,
                episode,
            )

        return {
            "correct": correct,
            "decision": decision,
            "answer": episode.answer_bit,
        }
