
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import math


@dataclass(frozen=True)
class OperatorEvidence:
    intervention_key: str
    predicted: int
    observed: int
    source: str = "intervention"


@dataclass(frozen=True)
class ExecutableOperator:
    operator_id: str
    expression: str
    input_names: Tuple[str, ...]
    output_type: str
    support: int
    contradictions: int
    complexity: float
    committed: bool = False

    @property
    def confidence(self):
        return (self.support + 1) / max(
            2, self.support + self.contradictions + 2
        )

    @property
    def score(self):
        return (
            math.log(max(1e-9, self.confidence))
            + 0.12 * self.support
            - 0.55 * self.contradictions
            - 0.12 * self.complexity
        )


class InducedOperatorSynthesizer:
    """
    Converts an induced schema into explicit executable operators.

    Critical rule:
        schema discovery != model commitment

    An induced schema first generates candidate executable expressions.
    Candidates are tested against accumulated intervention evidence. A model is
    committed only when it is both empirically supported and uniquely preferred.
    """

    def __init__(self, typed_generator):
        self.generator = typed_generator
        self.operators: Dict[str, ExecutableOperator] = {}
        self.history = []

    def _candidate_terms(self, schema, values):
        # Start from expressions supported by the induced output type.
        terms = list(
            self.generator.expand(
                values,
                {
                    k: (
                        "bit"
                        if isinstance(v, int) and v in (0, 1)
                        else "integer"
                    )
                    for k, v in values.items()
                },
                operator_library=None,
            )
        )

        if schema.output_type == "integer":
            # Explicitly construct the induced arithmetic family.
            from typed_language import Term

            vars_ = [
                Term(kind="var", name=k)
                for k, v in values.items()
                if isinstance(v, int)
            ]

            terms.extend(
                [
                    Term(kind="const", const=0),
                    Term(kind="const", const=1),
                    Term(kind="const", const=2),
                ]
            )

            for i, a in enumerate(vars_):
                for b in vars_[i + 1:]:
                    terms.append(
                        Term(kind="add", args=(a, b))
                    )

        unique = {}
        for term in terms:
            unique[term.text()] = term
        return tuple(unique.values())

    def induce(
        self,
        task,
        regime,
        schema,
        values,
    ):
        terms = self._candidate_terms(
            schema,
            values,
        )

        candidates = []
        for term in terms:
            op_id = (
                f"{task}|r{regime}|"
                f"verified|{term.text()}"
            )
            candidate = ExecutableOperator(
                operator_id=op_id,
                expression=term.text(),
                input_names=tuple(sorted(set(term.leaves()))),
                output_type=schema.output_type,
                support=0,
                contradictions=0,
                complexity=(
                    1.0
                    + 0.25 * term.depth()
                    + 0.08 * len(term.text())
                ),
            )
            self.operators[op_id] = candidate
            candidates.append(candidate)

        self.history.append(
            (
                "operator_candidates",
                task,
                regime,
                schema.schema_name,
                len(candidates),
            )
        )
        return tuple(candidates)

    def verify(
        self,
        candidates,
        evidence,
        values_by_intervention,
    ):
        """
        Score candidates solely against explicit intervention evidence.
        """
        updated = []

        for candidate in candidates:
            support = 0
            contradictions = 0

            # Reconstruct the expression from the typed term text through the
            # supplied candidate term object attached temporarily by caller.
            term = getattr(candidate, "_term", None)
            if term is None:
                continue

            for ev in evidence:
                local = dict(
                    values_by_intervention[
                        ev.intervention_key
                    ]
                )
                try:
                    pred = self.generator.execute(
                        term,
                        local,
                    )
                except Exception:
                    contradictions += 1
                    continue

                if pred == ev.observed:
                    support += 1
                else:
                    contradictions += 1

            new = ExecutableOperator(
                **{
                    **candidate.__dict__,
                    "support": support,
                    "contradictions": contradictions,
                }
            )
            self.operators[
                candidate.operator_id
            ] = new
            updated.append(new)

        return tuple(updated)

    def select(self, candidates, min_support=1):
        eligible = [
            c for c in candidates
            if c.support >= min_support
            and c.contradictions == 0
        ]

        if not eligible:
            return None

        ranked = sorted(
            eligible,
            key=lambda c: c.score,
            reverse=True,
        )

        if len(ranked) == 1:
            return ranked[0]

        # Require unique empirical agreement, then use structural simplicity.
        best, second = ranked[:2]

        if (
            best.support > second.support
            or best.complexity < second.complexity
        ):
            return best

        return None


def attach_terms(
    candidates,
    terms,
):
    by_text={
        t.text(): t
        for t in terms
    }
    out=[]
    for c in candidates:
        t=by_text.get(c.expression)
        if t is not None:
            object.__setattr__(c, "_term", t)
        out.append(c)
    return tuple(out)
