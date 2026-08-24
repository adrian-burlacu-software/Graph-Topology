from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random


@dataclass
class Genome:
    # Plasticity
    reuse_threshold: float = 0.55
    branch_threshold: float = 0.35

    # Designer learning
    learning_rate: float = 0.10
    learning_decay: float = 0.995
    reward_learning: float = 0.08
    punishment_learning: float = 0.05

    # Synapses
    initial_excitation: float = 1.0
    initial_inhibition: float = 0.6
    max_weight: float = 2.0
    min_weight: float = 0.05

    # Neuron dynamics
    firing_threshold: float = 1.0
    inhibition_decay: float = 0.90
    potential_decay: float = 0.85

    # Structural plasticity
    reuse_bonus: float = 1.0
    branch_cost: float = 0.25
    creation_cost: float = 0.10

    # Feedback
    feedback_strength: float = 0.20

    def clamp(self) -> "Genome":
        self.reuse_threshold = max(0.0, min(2.0, self.reuse_threshold))
        self.branch_threshold = max(-2.0, min(2.0, self.branch_threshold))

        self.learning_rate = max(0.001, min(1.0, self.learning_rate))
        self.learning_decay = max(0.90, min(1.0, self.learning_decay))
        self.reward_learning = max(0.0, min(1.0, self.reward_learning))
        self.punishment_learning = max(0.0, min(1.0, self.punishment_learning))

        self.initial_excitation = max(0.05, min(3.0, self.initial_excitation))
        self.initial_inhibition = max(0.05, min(3.0, self.initial_inhibition))
        self.max_weight = max(self.min_weight, min(5.0, self.max_weight))
        self.min_weight = max(0.0, min(1.0, self.min_weight))

        self.firing_threshold = max(0.1, min(5.0, self.firing_threshold))
        self.inhibition_decay = max(0.0, min(1.0, self.inhibition_decay))
        self.potential_decay = max(0.0, min(1.0, self.potential_decay))

        self.reuse_bonus = max(0.0, min(5.0, self.reuse_bonus))
        self.branch_cost = max(0.0, min(5.0, self.branch_cost))
        self.creation_cost = max(0.0, min(5.0, self.creation_cost))

        self.feedback_strength = max(0.0, min(2.0, self.feedback_strength))

        return self

    def mutate(
        self,
        rng: random.Random | None = None,
        rate: float = 0.15,
        scale: float = 0.15,
    ) -> "Genome":
        rng = rng or random.Random()

        values = asdict(self)

        for key, value in values.items():
            if rng.random() > rate:
                continue

            if isinstance(value, float):
                # Multiplicative mutation preserves useful scale.
                factor = 1.0 + rng.uniform(-scale, scale)
                values[key] = value * factor

        return Genome(**values).clamp()

    def crossover(
        self,
        other: "Genome",
        rng: random.Random | None = None,
    ) -> "Genome":
        rng = rng or random.Random()

        a = asdict(self)
        b = asdict(other)

        child = {}

        for key in a:
            if rng.random() < 0.5:
                child[key] = a[key]
            else:
                child[key] = b[key]

        return Genome(**child).clamp()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        fields = cls.__dataclass_fields__

        clean = {
            key: value
            for key, value in data.items()
            if key in fields
        }

        return cls(**clean).clamp()

    @classmethod
    def from_json(cls, text: str) -> "Genome":
        return cls.from_dict(json.loads(text))


# ------------------------------------------------------------
# EXPERIMENTAL GENOME
# ------------------------------------------------------------
#
# This is intentionally conservative.
#
# The important hypothesis:
#
#   reuse should become increasingly attractive when a matching
#   reusable vocabulary structure exists, while branch creation
#   should remain possible when no matching structure exists.
#
# Do NOT crank these values aggressively yet.
#

GENOME = Genome(
    reuse_threshold=0.55,
    branch_threshold=0.35,

    learning_rate=0.10,
    learning_decay=0.995,

    reward_learning=0.08,
    punishment_learning=0.05,

    initial_excitation=1.0,
    initial_inhibition=0.6,

    max_weight=2.0,
    min_weight=0.05,

    firing_threshold=1.0,

    inhibition_decay=0.90,
    potential_decay=0.85,

    reuse_bonus=1.0,
    branch_cost=0.25,
    creation_cost=0.10,

    feedback_strength=0.20,
)


def get_genome() -> Genome:
    """
    Return a fresh copy so an experiment cannot accidentally mutate
    the global genome definition.
    """
    return Genome.from_dict(GENOME.to_dict())


def mutate_genome(
    genome: Genome | None = None,
    seed: int | None = None,
    rate: float = 0.15,
    scale: float = 0.15,
) -> Genome:
    rng = random.Random(seed)

    if genome is None:
        genome = get_genome()

    return genome.mutate(
        rng=rng,
        rate=rate,
        scale=scale,
    )


if __name__ == "__main__":
    genome = get_genome()

    print("=== GENOME ===")
    print(genome.to_json())

    print("\n=== MUTATION ===")
    print(mutate_genome(genome, seed=42).to_json())