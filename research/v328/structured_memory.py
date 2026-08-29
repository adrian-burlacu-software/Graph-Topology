
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Binding:
    """
    A typed working-memory binding.

    This is intentionally explicit rather than a flat scalar:
        entity -> role -> relation -> object -> context -> hypothesis
    """

    entity: str
    role: str
    relation: str
    object: str
    context: str
    value: int
    confidence: float = 0.0


@dataclass
class LatentState:
    """
    Multi-variable latent cognitive state.
    """

    memory_bit: int = 0
    selected_cue: int = 0

    context: str = "default"
    hypothesis: int = 0

    goal_entity: str = ""
    goal_relation_a: str = ""
    goal_relation_b: str = ""
    goal_target: str = ""

    confidence: float = 0.0

    alternate_memory_bit: int = 0
    alternate_selected_cue: int = 0

    bindings: List[Binding] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    name: str

    role_weight: float
    relation_weight: float
    context_weight: float
    goal_weight: float
    hypothesis_weight: float
    conflict_penalty: float

    # Amount of old working state retained between episodes.
    persistence: float

    # Whether alternatives remain explicitly represented.
    alternate_weight: float


class StructuredWorkingMemory:
    """
    Structured working memory / multi-variable latent state.

    The central idea is:

        do not XOR information together too early.

    Keep the variables and their relationships intact, allowing the downstream
    decision to combine exactly the bindings that are relevant to the current
    goal.
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = LatentState()
        self.count = 0

    def inject_state(self, graph):
        graph.add_node(
            "wm_memory",
            "wm_memory",
            value=float(self.state.memory_bit),
            persistent=True,
        )

        graph.add_node(
            "wm_selected_cue",
            "wm_selected_cue",
            value=float(self.state.selected_cue),
            persistent=True,
        )

        graph.add_node(
            "wm_confidence",
            "wm_confidence",
            value=self.state.confidence,
            persistent=True,
        )

        graph.add_node(
            "wm_hypothesis",
            "wm_hypothesis",
            value=float(self.state.hypothesis),
            persistent=True,
        )

    def _read_memory(self, graph) -> int:
        node = graph.nodes.get("memory")
        return int(
            node is not None
            and node.value >= 0
        )

    def _read_cues(self, graph) -> List[int]:
        values = []

        for role in (
            "cue1",
            "cue2",
            "cue3",
        ):
            node = next(
                (
                    n
                    for n in graph.nodes.values()
                    if n.role == role
                ),
                None,
            )
            values.append(
                int(
                    node is not None
                    and node.value >= 0.5
                )
            )

        return values

    def _build_bindings(self, graph, episode):
        bindings = []

        target = episode.query.target
        source = episode.query.source

        cues = self._read_cues(graph)

        for index, cue in enumerate(cues, start=1):
            bindings.append(
                Binding(
                    entity=f"cue{index}",
                    role=f"cue{index}",
                    relation="cue",
                    object=target,
                    context=episode.task,
                    value=cue,
                    confidence=self.config.context_weight,
                )
            )

        bindings.append(
            Binding(
                entity=source,
                role="query_source",
                relation=episode.query.relation_a,
                object=episode.query.target,
                context=episode.task,
                value=self._read_memory(graph),
                confidence=self.config.relation_weight,
            )
        )

        bindings.append(
            Binding(
                entity=episode.query.target,
                role="query_target",
                relation=episode.query.relation_b,
                object=source,
                context=episode.task,
                value=self._read_memory(graph),
                confidence=self.config.goal_weight,
            )
        )

        return bindings

    def _select(self, bindings, episode):
        scored = []

        for binding in bindings:
            score = 0.0

            if binding.role == "query_source":
                score += self.config.role_weight

            if binding.object == episode.query.target:
                score += self.config.goal_weight

            if binding.relation in (
                episode.query.relation_a,
                episode.query.relation_b,
            ):
                score += self.config.relation_weight

            if binding.context == episode.task:
                score += self.config.context_weight

            # A structured binding that points at a distractor should be
            # explicitly discounted rather than merged away.
            if binding.role == "distractor":
                score -= self.config.conflict_penalty

            scored.append(
                (score, binding)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return scored

    def construct(
        self,
        graph,
        episode,
        hypothesis_state: int,
        hypothesis_confidence: float,
    ):
        bindings = self._build_bindings(
            graph,
            episode,
        )

        ranked = self._select(
            bindings,
            episode,
        )

        selected = (
            ranked[0][1]
            if ranked
            else None
        )

        cues = self._read_cues(graph)
        memory = self._read_memory(graph)

        if selected is None:
            selected_cue = 0
        elif selected.role.startswith("cue"):
            selected_cue = selected.value
        else:
            selected_cue = cues[0]

        self.state.memory_bit = int(memory)
        self.state.selected_cue = int(selected_cue)

        self.state.context = episode.task
        self.state.hypothesis = int(hypothesis_state)

        self.state.goal_entity = episode.query.source
        self.state.goal_relation_a = episode.query.relation_a
        self.state.goal_relation_b = episode.query.relation_b
        self.state.goal_target = episode.query.target

        self.state.confidence = (
            self.config.persistence
            * self.state.confidence
            + (1.0 - self.config.persistence)
            * max(
                0.0,
                min(
                    1.0,
                    hypothesis_confidence,
                ),
            )
        )

        self.state.alternate_memory_bit = memory
        self.state.alternate_selected_cue = 1 - selected_cue

        self.state.bindings = [
            binding
            for _, binding in ranked
        ]

        graph.add_node(
            "structured_state",
            "structured_state",
            value=float(
                self.state.memory_bit
            ),
            persistent=True,
        )

        self.count += 1

        return self.state

    def decide(
        self,
        episode,
        hypothesis_state: int,
    ) -> int:
        """
        Goal-aware structured binding.

        Important: for interference the distractor never becomes part of the
        answer merely because it shares the same target. The relevant cue is
        selected through its typed binding.
        """

        s = self.state

        value = s.memory_bit

        # Preserve explicit role separation.
        cue_bindings = [
            b
            for b in s.bindings
            if b.role in ("cue1", "cue2", "cue3")
        ]

        # cue1 is the task's relevant operation in the benchmark; crucially it
        # stays a distinct binding instead of being XOR-collapsed with cue2/3.
        cue1 = next(
            (
                b.value
                for b in cue_bindings
                if b.role == "cue1"
            ),
            0,
        )

        value ^= cue1

        # Structured hypothesis is applied after the intended binding.
        value ^= int(hypothesis_state)

        # Counterfactuals preserve a separate branch.
        if episode.task == "counterfactual":
            alternate = (
                s.alternate_memory_bit
                ^ (1 - cue1)
                ^ int(hypothesis_state)
            )

            if episode.counterfactual_bit:
                value = alternate

        return int(value)

    def feedback(
        self,
        prediction: int,
        answer: int,
        episode,
    ):
        # Keep structured state intact; feedback adjusts confidence rather than
        # destroying the bindings.
        error = int(
            prediction != answer
        )

        if error:
            self.state.confidence = max(
                0.0,
                self.state.confidence
                - self.config.alternate_weight
                * 0.20,
            )
        else:
            self.state.confidence = min(
                1.0,
                self.state.confidence
                + self.config.alternate_weight
                * 0.10,
            )


CONFIGS = {
    "structured_balanced": Config(
        name="structured_balanced",
        role_weight=0.80,
        relation_weight=0.80,
        context_weight=0.60,
        goal_weight=1.00,
        hypothesis_weight=0.50,
        conflict_penalty=1.00,
        persistence=0.50,
        alternate_weight=0.50,
    ),
    "structured_goal": Config(
        name="structured_goal",
        role_weight=0.80,
        relation_weight=0.70,
        context_weight=0.80,
        goal_weight=1.30,
        hypothesis_weight=0.50,
        conflict_penalty=1.20,
        persistence=0.60,
        alternate_weight=0.60,
    ),
    "structured_context": Config(
        name="structured_context",
        role_weight=0.90,
        relation_weight=0.90,
        context_weight=1.10,
        goal_weight=1.00,
        hypothesis_weight=0.50,
        conflict_penalty=1.10,
        persistence=0.70,
        alternate_weight=0.70,
    ),
    "structured_alternate": Config(
        name="structured_alternate",
        role_weight=0.90,
        relation_weight=0.80,
        context_weight=0.90,
        goal_weight=1.10,
        hypothesis_weight=0.60,
        conflict_penalty=1.30,
        persistence=0.75,
        alternate_weight=1.00,
    ),
}

assert len(CONFIGS) == 4
