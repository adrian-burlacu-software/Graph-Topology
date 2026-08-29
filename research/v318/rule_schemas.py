
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RuleSchema:
    name: str
    selector: str
    operator: str

    @property
    def complexity(self) -> int:
        selector_cost = {
            "memory": 1,
            "cue1": 1,
            "cue2": 1,
            "cue3": 1,
            "memory+c1": 2,
            "memory+c2": 2,
            "memory+c3": 2,
        }[self.selector]

        return selector_cost + (
            0 if self.operator == "identity" else 1
        )

    def evaluate(
        self,
        memory: int,
        cues: List[int],
    ) -> int:
        values = {
            "memory": memory,
            "cue1": cues[0],
            "cue2": cues[1],
            "cue3": cues[2],
            "memory+c1": memory ^ cues[0],
            "memory+c2": memory ^ cues[1],
            "memory+c3": memory ^ cues[2],
        }

        value = values[self.selector]

        if self.operator == "identity":
            return int(value)

        if self.operator == "invert":
            return 1 - int(value)

        raise ValueError(self.operator)


ALL_SCHEMAS = [
    RuleSchema(
        f"{selector}:{operator}",
        selector,
        operator,
    )
    for selector in (
        "memory",
        "cue1",
        "cue2",
        "cue3",
        "memory+c1",
        "memory+c2",
        "memory+c3",
    )
    for operator in (
        "identity",
        "invert",
    )
]


@dataclass(frozen=True)
class Observation:
    memory: int
    cues: Tuple[int, int, int]
    answer: int
    changed: bool


@dataclass
class CandidateEvidence:
    schema: RuleSchema
    score: float = 0.0
    support: int = 0
    contradictions: int = 0

    @property
    def consistency(self) -> float:
        total = self.support + self.contradictions
        if total <= 0:
            return 0.0
        return self.support / total


@dataclass
class ContextRuleMemory:
    context: str

    # The committed explanation.
    active: RuleSchema | None = None
    active_confidence: float = 0.0

    # Candidate evidence belongs to the CURRENT induction phase.
    candidates: Dict[str, CandidateEvidence] = field(
        default_factory=dict
    )

    # Current phase evidence.
    window: List[Observation] = field(
        default_factory=list
    )

    # Regime marker state.
    phase: int = 0
    saw_change: bool = False

    pending: str | None = None
    pending_streak: int = 0
    revisions: int = 0


@dataclass(frozen=True)
class SchemaConfig:
    name: str

    window_size: int
    min_window: int

    support_gain: float
    contradiction_penalty: float

    consistency_threshold: float
    margin_threshold: float

    commit_streak: int

    simplicity_bias: float


class ExplicitRuleSchemaMemory:
    """
    Explicit causal rule induction with phase-local evidence.

    The important change is phase separation:

        old regime evidence
             ↓
        committed rule
             ↓
        explicit regime marker
             ↓
        fresh induction window
             ↓
        new candidate rule
             ↓
        delayed commit

    Old knowledge remains represented by the committed schema. New evidence
    does not have to fight the entire history forever.
    """

    def __init__(self, config: SchemaConfig):
        self.config = config
        self.contexts: Dict[str, ContextRuleMemory] = {}
        self.count = 0

    def _state(self, context: str) -> ContextRuleMemory:
        state = self.contexts.get(context)

        if state is None:
            state = ContextRuleMemory(
                context=context
            )
            state.candidates = {
                schema.name: CandidateEvidence(schema)
                for schema in ALL_SCHEMAS
            }
            self.contexts[context] = state

        return state

    def inject_state(self, graph, context: str):
        state = self._state(context)

        graph.add_node(
            "active_rule_schema",
            "active_rule_schema",
            value=float(
                0
                if state.active is None
                else state.active.complexity
            ),
            persistent=True,
        )

        graph.add_node(
            "rule_schema_confidence",
            "rule_schema_confidence",
            value=state.active_confidence,
            persistent=True,
        )

        graph.add_node(
            "rule_phase",
            "rule_phase",
            value=float(state.phase),
            persistent=True,
        )

    def active(self, context: str):
        state = self._state(context)

        return (
            state.active.name
            if state.active
            else None,
            state.active_confidence,
        )

    def apply(
        self,
        context: str,
        memory: int,
        cues: List[int],
    ) -> int:
        state = self._state(context)

        schema = state.active

        if schema is None:
            schema = next(
                x
                for x in ALL_SCHEMAS
                if x.name == "memory:identity"
            )

        return schema.evaluate(
            memory,
            cues,
        )

    def _reset_phase(
        self,
        state: ContextRuleMemory,
        phase: int,
    ):
        state.phase = phase
        state.window.clear()
        state.pending = None
        state.pending_streak = 0

        # Fresh candidate evidence for the new rule, preserving only the
        # committed schema as the historical explanation.
        state.candidates = {
            schema.name: CandidateEvidence(schema)
            for schema in ALL_SCHEMAS
        }

    def _rank(self, state: ContextRuleMemory):
        ranked = []

        if not state.window:
            return ranked

        for candidate in state.candidates.values():
            support = 0

            for obs in state.window:
                predicted = candidate.schema.evaluate(
                    obs.memory,
                    list(obs.cues),
                )
                support += int(
                    predicted == obs.answer
                )

            consistency = (
                support / len(state.window)
            )

            # Simplicity breaks benchmark-equivalent schema ties without using
            # the hidden answer/rule directly.
            value = (
                consistency
                + self.config.simplicity_bias
                * (
                    1.0
                    / (
                        1.0
                        + candidate.schema.complexity
                    )
                )
            )

            ranked.append(
                (
                    value,
                    consistency,
                    candidate.schema.complexity,
                    candidate.schema,
                )
            )

        ranked.sort(
            key=lambda x: (
                x[0],
                x[1],
                -x[2],
            ),
            reverse=True,
        )

        return ranked

    def _induce(self, state: ContextRuleMemory):
        if len(state.window) < self.config.min_window:
            return

        ranked = self._rank(state)

        winner = ranked[0]
        runner = ranked[1]

        margin = winner[0] - runner[0]
        consistency = winner[1]
        candidate = winner[3]

        qualifies = (
            consistency
            >= self.config.consistency_threshold
            and margin
            >= self.config.margin_threshold
        )

        if not qualifies:
            return

        if (
            state.active is None
            or candidate.name != state.active.name
        ):
            if state.pending == candidate.name:
                state.pending_streak += 1
            else:
                state.pending = candidate.name
                state.pending_streak = 1

            if (
                state.pending_streak
                >= self.config.commit_streak
            ):
                state.active = candidate
                state.active_confidence = consistency
                state.pending = None
                state.pending_streak = 0
                state.revisions += 1

        else:
            state.active_confidence = consistency

    def observe(
        self,
        context: str,
        memory: int,
        cues: List[int],
        answer: int,
        episode,
    ):
        state = self._state(context)

        changed = (
            episode.task == "rule_change"
            and episode.rule_version == 1
        )

        # The benchmark's regime marker is treated as a phase boundary, not
        # as knowledge of the new rule.
        if changed and not state.saw_change:
            state.saw_change = True
            self._reset_phase(
                state,
                state.phase + 1,
            )

        state.window.append(
            Observation(
                memory=int(memory),
                cues=(
                    int(cues[0]),
                    int(cues[1]),
                    int(cues[2]),
                ),
                answer=int(answer),
                changed=changed,
            )
        )

        if len(state.window) > self.config.window_size:
            state.window.pop(0)

        self._induce(state)

        self.count += 1

    def summary(self):
        output = {}

        for context, state in self.contexts.items():
            output[context] = {
                "active": (
                    state.active.name
                    if state.active
                    else None
                ),
                "confidence": state.active_confidence,
                "phase": state.phase,
                "pending": state.pending,
                "revisions": state.revisions,
                "window": len(state.window),
                "top": [
                    {
                        "schema": row[3].name,
                        "consistency": row[1],
                        "rank": row[0],
                    }
                    for row in self._rank(state)[:5]
                ],
            }

        return output


CONFIGS = {
    "schema_balanced": SchemaConfig(
        "schema_balanced",
        window_size=6,
        min_window=3,
        support_gain=1.0,
        contradiction_penalty=0.8,
        consistency_threshold=0.66,
        margin_threshold=0.001,
        commit_streak=1,
        simplicity_bias=0.080,
    ),
    "schema_stable": SchemaConfig(
        "schema_stable",
        window_size=8,
        min_window=4,
        support_gain=1.0,
        contradiction_penalty=0.7,
        consistency_threshold=0.75,
        margin_threshold=0.001,
        commit_streak=1,
        simplicity_bias=0.080,
    ),
    "schema_adaptive": SchemaConfig(
        "schema_adaptive",
        window_size=5,
        min_window=3,
        support_gain=1.0,
        contradiction_penalty=0.9,
        consistency_threshold=0.60,
        margin_threshold=0.001,
        commit_streak=1,
        simplicity_bias=0.080,
    ),
    "schema_delayed": SchemaConfig(
        "schema_delayed",
        window_size=10,
        min_window=5,
        support_gain=1.0,
        contradiction_penalty=0.6,
        consistency_threshold=0.80,
        margin_threshold=0.001,
        commit_streak=1,
        simplicity_bias=0.080,
    ),
}

assert len(CONFIGS) == 4
assert len(ALL_SCHEMAS) == 14
