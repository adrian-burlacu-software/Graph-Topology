
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


ACTIONS = ("LEFT", "RIGHT", "COMMIT", "REJECT")


@dataclass
class GNode:
    name: str
    activation: float = 0.0
    persistent: bool = False
    tag: str = ""


@dataclass
class GEdge:
    source: str
    relation: str
    target: str
    weight: float = 1.0
    persistent: bool = False


@dataclass
class CognitiveGraph:
    nodes: Dict[str, GNode] = field(default_factory=dict)
    edges: List[GEdge] = field(default_factory=list)

    def clone(self) -> "CognitiveGraph":
        return CognitiveGraph(
            nodes={
                k: GNode(
                    v.name,
                    v.activation,
                    v.persistent,
                    v.tag,
                )
                for k, v in self.nodes.items()
            },
            edges=[
                GEdge(
                    e.source,
                    e.relation,
                    e.target,
                    e.weight,
                    e.persistent,
                )
                for e in self.edges
            ],
        )

    def ensure_node(
        self,
        name: str,
        activation: float = 0.0,
        persistent: bool = False,
        tag: str = "",
    ) -> GNode:
        node = self.nodes.get(name)
        if node is None:
            node = GNode(
                name=name,
                activation=activation,
                persistent=persistent,
                tag=tag,
            )
            self.nodes[name] = node
        else:
            node.activation = max(
                node.activation,
                activation,
            )
            node.persistent = (
                node.persistent or persistent
            )
            if tag:
                node.tag = tag
        return node

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        persistent: bool = False,
    ) -> None:
        self.ensure_node(source)
        self.ensure_node(target)

        for edge in self.edges:
            if (
                edge.source == source
                and edge.relation == relation
                and edge.target == target
            ):
                edge.weight = max(
                    edge.weight,
                    weight,
                )
                edge.persistent = (
                    edge.persistent or persistent
                )
                return

        self.edges.append(
            GEdge(
                source,
                relation,
                target,
                weight,
                persistent,
            )
        )

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> bool:
        return any(
            e.source == source
            and e.relation == relation
            and e.target == target
            and e.weight > 0.0
            for e in self.edges
        )

    def incoming(
        self,
        target: str,
        relation: str | None = None,
    ) -> List[GEdge]:
        return [
            e
            for e in self.edges
            if e.target == target
            and (
                relation is None
                or e.relation == relation
            )
        ]

    def outgoing(
        self,
        source: str,
        relation: str | None = None,
    ) -> List[GEdge]:
        return [
            e
            for e in self.edges
            if e.source == source
            and (
                relation is None
                or e.relation == relation
            )
        ]

    def activate(
        self,
        name: str,
        amount: float,
    ) -> None:
        self.ensure_node(name)
        self.nodes[name].activation += amount

    def decay(
        self,
        factor: float,
        persistent_floor: float = 0.0,
    ) -> None:
        for node in self.nodes.values():
            if node.persistent:
                node.activation = max(
                    persistent_floor,
                    node.activation * factor,
                )
            else:
                node.activation *= factor

    def commit_persistent(self) -> None:
        for node in self.nodes.values():
            if node.activation > 0.5:
                node.persistent = True
        for edge in self.edges:
            if edge.weight > 0.5:
                edge.persistent = True

    def signature(self) -> Tuple:
        return (
            tuple(
                sorted(
                    (
                        n.name,
                        round(n.activation, 4),
                        n.persistent,
                        n.tag,
                    )
                    for n in self.nodes.values()
                )
            ),
            tuple(
                sorted(
                    (
                        e.source,
                        e.relation,
                        e.target,
                        round(e.weight, 4),
                        e.persistent,
                    )
                    for e in self.edges
                )
            ),
        )


@dataclass(frozen=True)
class Task:
    seed: int
    task_type: str
    hidden_bit: int
    distractor: int
    terminal_bit: int
    expected_action: str


TASK_TYPES = (
    "delayed_recall",
    "interference",
    "composition",
    "counterfactual",
)


def make_task(
    seed: int,
    task_type: str,
    hidden_bit: int,
) -> Task:
    # Deterministic task generator. All task types are tiny so the entire
    # search remains CPU-cheap.
    distractor = (
        (seed * 7 + hidden_bit * 3) % 2
    )
    terminal_bit = (
        (seed * 11 + hidden_bit) % 2
    )

    if task_type == "delayed_recall":
        result = hidden_bit

    elif task_type == "interference":
        # The distractor is intentionally different from the memory.
        result = hidden_bit

    elif task_type == "composition":
        # Apply XOR composition to hidden information and terminal cue.
        result = hidden_bit ^ terminal_bit

    elif task_type == "counterfactual":
        # Terminal cue tells the controller whether to preserve or invert the
        # earlier fact.
        result = (
            hidden_bit
            if terminal_bit == 0
            else 1 - hidden_bit
        )

    else:
        raise ValueError(task_type)

    expected = (
        "LEFT"
        if result == 0
        else "RIGHT"
    )

    return Task(
        seed=seed,
        task_type=task_type,
        hidden_bit=hidden_bit,
        distractor=distractor,
        terminal_bit=terminal_bit,
        expected_action=expected,
    )


class MemoryModule:
    name = "base"

    def write(
        self,
        graph: CognitiveGraph,
        task: Task,
    ) -> None:
        raise NotImplementedError

    def maintain(
        self,
        graph: CognitiveGraph,
        task: Task,
        step: int,
    ) -> None:
        return None


class EdgeMemory(MemoryModule):
    name = "edge"

    def write(self, graph, task):
        token = "MEM_ONE" if task.hidden_bit else "MEM_ZERO"
        graph.add_edge(
            "MEMORY",
            "stores",
            token,
            weight=1.0,
            persistent=True,
        )


class ActivationMemory(MemoryModule):
    name = "activation"

    def write(self, graph, task):
        graph.activate(
            "MEMORY_ONE" if task.hidden_bit else "MEMORY_ZERO",
            1.0,
        )


class PersistentSlotMemory(MemoryModule):
    name = "persistent_slot"

    def write(self, graph, task):
        graph.ensure_node(
            "MEMORY_SLOT",
            activation=1.0 if task.hidden_bit else 0.25,
            persistent=True,
            tag=str(task.hidden_bit),
        )

    def maintain(self, graph, task, step):
        graph.activate(
            "MEMORY_SLOT",
            0.05,
        )


class CreditModule:
    name = "none"

    def apply(
        self,
        graph: CognitiveGraph,
        task: Task,
        correct: bool,
        step: int,
    ) -> None:
        return None


class ImmediateCredit(CreditModule):
    name = "immediate"

    def apply(self, graph, task, correct, step):
        graph.activate(
            "CREDIT_CORRECT"
            if correct
            else "CREDIT_WRONG",
            0.2 if correct else -0.1,
        )


class EligibilityCredit(CreditModule):
    name = "eligibility"

    def apply(self, graph, task, correct, step):
        amount = 0.35 if correct else -0.15
        graph.activate(
            "ELIGIBILITY",
            amount,
        )
        graph.add_edge(
            "ELIGIBILITY",
            "targets",
            "MEMORY_TRACE",
            weight=max(0.1, amount),
        )


class TDValueCredit(CreditModule):
    name = "td"

    def apply(self, graph, task, correct, step):
        target = 0.4 if correct else -0.2
        graph.activate(
            "VALUE",
            target,
        )


class DynamicsModule:
    name = "static"

    def pre_step(
        self,
        graph: CognitiveGraph,
        step: int,
    ) -> None:
        return None

    def post_step(
        self,
        graph: CognitiveGraph,
        step: int,
    ) -> None:
        return None


class LeakyDynamics(DynamicsModule):
    name = "leaky"

    def pre_step(self, graph, step):
        graph.decay(0.85)

    def post_step(self, graph, step):
        graph.decay(0.95)


class PersistentDynamics(DynamicsModule):
    name = "persistent"

    def pre_step(self, graph, step):
        graph.decay(
            0.80,
            persistent_floor=0.60,
        )

    def post_step(self, graph, step):
        graph.decay(
            0.95,
            persistent_floor=0.60,
        )


class GatedDynamics(DynamicsModule):
    name = "gated"

    def pre_step(self, graph, step):
        graph.decay(0.80)

        if step >= 1:
            for node in graph.nodes.values():
                if node.persistent:
                    node.activation = max(
                        node.activation,
                        0.75,
                    )

    def post_step(self, graph, step):
        graph.decay(0.97)


class ReadoutModule:
    name = "structural"

    def read(
        self,
        graph: CognitiveGraph,
        task: Task,
    ) -> int:
        raise NotImplementedError


class StructuralReadout(ReadoutModule):
    name = "structural"

    def read(self, graph, task):
        token = (
            "MEM_ONE"
            if graph.has_edge(
                "MEMORY",
                "stores",
                "MEM_ONE",
            )
            else "MEM_ZERO"
        )
        return 1 if token == "MEM_ONE" else 0


class ActivationReadout(ReadoutModule):
    name = "activation"

    def read(self, graph, task):
        one = graph.nodes.get(
            "MEMORY_ONE",
            GNode("x"),
        ).activation
        zero = graph.nodes.get(
            "MEMORY_ZERO",
            GNode("x"),
        ).activation
        return 1 if one >= zero else 0


class VotingReadout(ReadoutModule):
    name = "voting"

    def read(self, graph, task):
        score = 0.0

        for edge in graph.outgoing("MEMORY"):
            if edge.target == "MEM_ONE":
                score += edge.weight
            elif edge.target == "MEM_ZERO":
                score -= edge.weight

        slot = graph.nodes.get("MEMORY_SLOT")
        if slot is not None:
            score += (
                0.5
                if slot.tag == "1"
                else -0.5
            ) * max(
                0.0,
                slot.activation,
            )

        score += graph.nodes.get(
            "CREDIT_CORRECT",
            GNode("x"),
        ).activation

        score -= graph.nodes.get(
            "CREDIT_WRONG",
            GNode("x"),
        ).activation

        return 1 if score >= 0 else 0


class PlanningModule:
    name = "none"

    def transform(
        self,
        graph: CognitiveGraph,
        task: Task,
        recalled_bit: int,
    ) -> int:
        return recalled_bit


class OneStepComposition(PlanningModule):
    name = "one_step"

    def transform(self, graph, task, recalled_bit):
        if task.task_type == "composition":
            return recalled_bit ^ task.terminal_bit
        if task.task_type == "counterfactual":
            return (
                recalled_bit
                if task.terminal_bit == 0
                else 1 - recalled_bit
            )
        return recalled_bit


class TwoStepPlanning(PlanningModule):
    name = "two_step"

    def transform(self, graph, task, recalled_bit):
        bit = recalled_bit

        # First operation: combine with terminal cue.
        if task.task_type == "composition":
            bit ^= task.terminal_bit

        # Second operation: counterfactual inversion only when requested.
        if task.task_type == "counterfactual":
            if task.terminal_bit:
                bit = 1 - bit

        return bit


MEMORY = {
    "edge": EdgeMemory,
    "activation": ActivationMemory,
    "persistent_slot": PersistentSlotMemory,
}

CREDIT = {
    "none": CreditModule,
    "immediate": ImmediateCredit,
    "eligibility": EligibilityCredit,
    "td": TDValueCredit,
}

DYNAMICS = {
    "static": DynamicsModule,
    "leaky": LeakyDynamics,
    "persistent": PersistentDynamics,
    "gated": GatedDynamics,
}

READOUT = {
    "structural": StructuralReadout,
    "activation": ActivationReadout,
    "voting": VotingReadout,
}

PLANNING = {
    "none": PlanningModule,
    "one_step": OneStepComposition,
    "two_step": TwoStepPlanning,
}


@dataclass(frozen=True)
class Strategy:
    memory: str
    credit: str
    dynamics: str
    readout: str
    planning: str

    @property
    def name(self) -> str:
        return (
            f"{self.memory}+{self.credit}+"
            f"{self.dynamics}+{self.readout}+"
            f"{self.planning}"
        )


class CognitiveAlgorithm:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.memory = MEMORY[strategy.memory]()
        self.credit = CREDIT[strategy.credit]()
        self.dynamics = DYNAMICS[strategy.dynamics]()
        self.readout = READOUT[strategy.readout]()
        self.planning = PLANNING[strategy.planning]()

    def run(
        self,
        task: Task,
        horizon: int = 4,
        trace: bool = False,
    ) -> dict:
        graph = CognitiveGraph()

        graph.ensure_node("MEMORY")
        graph.ensure_node("GOAL")

        self.memory.write(
            graph,
            task,
        )

        if trace:
            trace_rows = []
        else:
            trace_rows = None

        # H-1 neutral/distractor transitions.
        for step in range(1, horizon):
            self.dynamics.pre_step(
                graph,
                step,
            )

            # Interference exists in the graph but is explicitly tagged as
            # distractor state. A cognitive algorithm must avoid replacing its
            # memory with this unrelated cue.
            graph.ensure_node(
                f"DISTRACTOR_{step}",
                activation=(
                    1.0
                    if task.distractor == (step % 2)
                    else 0.2
                ),
                tag="distractor",
            )
            graph.add_edge(
                "GOAL",
                "distractor",
                f"DISTRACTOR_{step}",
                weight=0.5,
            )

            self.memory.maintain(
                graph,
                task,
                step,
            )

            self.dynamics.post_step(
                graph,
                step,
            )

            if trace:
                trace_rows.append(
                    {
                        "step": step,
                        "memory_signature": graph.signature(),
                    }
                )

        recalled = self.readout.read(
            graph,
            task,
        )

        predicted_bit = self.planning.transform(
            graph,
            task,
            recalled,
        )

        action = (
            "LEFT"
            if predicted_bit == 0
            else "RIGHT"
        )

        correct = action == task.expected_action

        # Credit is applied after the terminal decision. For search, the
        # learning modules are deliberately tiny and deterministic.
        self.credit.apply(
            graph,
            task,
            correct,
            horizon,
        )

        if trace:
            return {
                "correct": correct,
                "action": action,
                "expected": task.expected_action,
                "recalled_bit": recalled,
                "trace": trace_rows,
                "graph": graph,
            }

        return {
            "correct": correct,
            "action": action,
            "expected": task.expected_action,
            "recalled_bit": recalled,
        }


def all_strategies() -> List[Strategy]:
    return [
        Strategy(
            memory,
            credit,
            dynamics,
            readout,
            planning,
        )
        for memory in MEMORY
        for credit in CREDIT
        for dynamics in DYNAMICS
        for readout in READOUT
        for planning in PLANNING
    ]
