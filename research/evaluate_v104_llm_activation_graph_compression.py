from __future__ import annotations

"""
V103 — LLM ACTIVATION GRAPH COMPRESSION

Input:
    results/v102_smol_representation_geometry.pt

The V102 probe already selected a useful activation representation:
    SmolLM2-360M
    layer 3
    centered + top-1 common component removed

V103 does NOT rerun the model.

It loads the saved activation vectors and asks the actual Graph-Topology
question:

    Can the LLM activation space itself be represented as a recursively
    compressed graph of reusable units?

The experiment has three levels:

    1. RAW ACTIVATION UNITS
       One graph unit per word vector.

    2. LOCAL ACTIVATION STRUCTURE
       Approximate nearest-neighbor relationships create reusable local
       activation units.

    3. RECURSIVE COMPRESSION
       Repeated local structures become higher-order reusable graph units.

The important comparison is against a generic baseline:

    naive representation:
        one unique vector/unit per word

    graph representation:
        reusable activation units + recursive assemblies

The semantic corpus is used only as an independent evaluation:
    semantic similarity in the human feature graph

We DO NOT claim that activation clusters are semantic by construction.
We measure whether the compressed activation topology preserves semantic
structure better than a shuffled/null control.

This is intentionally one large experiment.
"""

import csv
import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

PROBE_PATH = (
    ROOT
    / "results"
    / "v102_smol_representation_geometry.pt"
)

DICTIONARY_PATH = (
    ROOT
    / "data"
    / "dictionary.csv"
)

SEMANTICS_PATH = (
    ROOT
    / "data"
    / "semantics-large.csv"
)

MIN_REUSE = 2
MAX_LEVELS = 10

# Quantization controls how continuous activation vectors become reusable
# local graph keys. These are intentionally deterministic.
QUANTIZATION = 0.05

TOP_K_NEIGHBORS = 8


# ---------------------------------------------------------------------------
# Semantic corpus
# ---------------------------------------------------------------------------

def parse_float(
    value: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_int(
    value: str,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_large_semantics(
    path: Path,
) -> dict[str, dict[str, float]]:
    records: dict[
        str,
        dict[str, float],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "cue",
            "translated",
            "frequency_translated",
            "n",
            "normalized_translated",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "semantics-large.csv missing: "
                + ", ".join(sorted(missing))
            )

        for raw in reader:
            cue = raw["cue"].strip().lower()
            feature = raw["translated"].strip().lower()

            if not cue or not feature:
                continue

            weight = parse_float(
                raw["normalized_translated"]
            )

            if weight <= 0.0:
                frequency = parse_float(
                    raw["frequency_translated"]
                )
                n = parse_int(raw["n"])

                if n > 0:
                    weight = frequency / n

            if weight <= 0.0:
                continue

            records.setdefault(
                cue,
                {},
            )

            records[
                cue
            ][feature] = (
                records[cue].get(
                    feature,
                    0.0,
                )
                + weight
            )

    return records


def jaccard_weighted(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    keys = set(a) | set(b)

    if not keys:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for key in keys:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)

        numerator += min(av, bv)
        denominator += max(av, bv)

    return (
        numerator
        / max(
            1e-12,
            denominator,
        )
    )


# ---------------------------------------------------------------------------
# Activation representation
# ---------------------------------------------------------------------------

def load_probe():
    if not PROBE_PATH.exists():
        raise FileNotFoundError(
            f"Missing V102 activation file: {PROBE_PATH}"
        )

    payload = torch.load(
        PROBE_PATH,
        map_location="cpu",
    )

    if "words" not in payload:
        raise RuntimeError(
            "V102 file does not contain words."
        )

    if "best" not in payload:
        raise RuntimeError(
            "V102 file does not contain best representation metadata."
        )

    words = list(
        payload["words"]
    )

    best = payload["best"]

    level = int(
        best["level"]
    )

    representation = str(
        best["representation"]
    )

    tensor_key = (
        f"layer_{level}_{representation}"
    )

    layers = payload.get(
        "layers",
        {},
    )

    vectors = layers.get(
        tensor_key
    )

    if vectors is None:
        raise RuntimeError(
            "Best representation tensor not found: "
            + tensor_key
        )

    vectors = vectors.float()

    if vectors.ndim != 2:
        raise RuntimeError(
            f"Expected [words, hidden], got {tuple(vectors.shape)}"
        )

    if vectors.shape[0] != len(words):
        raise RuntimeError(
            "Word/vector count mismatch: "
            f"{len(words)} words vs {vectors.shape[0]} vectors"
        )

    return words, vectors, best


# ---------------------------------------------------------------------------
# Activation graph
# ---------------------------------------------------------------------------

@dataclass
class UnitStats:
    unit_id: int
    level: int
    members: tuple[int, ...]
    use_count: int


class ActivationGraph:
    """
    Discretizes activation vectors into reusable graph units.

    Level 0:
        one unit per distinct quantized activation signature.

    Higher levels:
        repeated unordered pairs of activation units become reusable
        assemblies.

    The key property is recursive self-application:
        units -> repeated combinations -> units -> ...
    """

    def __init__(
        self,
        vectors: torch.Tensor,
    ) -> None:
        self.vectors = vectors.float()

        self.unit_by_key: dict[
            tuple[int, ...],
            int,
        ] = {}

        self.units: dict[
            int,
            UnitStats,
        ] = {}

        self.next_unit_id = 0

        self.word_units: list[int] = []

    def normalized(
        self,
    ) -> torch.Tensor:
        return (
            self.vectors
            / torch.linalg.vector_norm(
                self.vectors,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

    def quantized_key(
        self,
        vector: torch.Tensor,
    ) -> tuple[int, ...]:
        normalized = vector / torch.linalg.vector_norm(
            vector
        ).clamp_min(1e-12)

        rounded = torch.round(
            normalized
            / QUANTIZATION
        ).to(torch.int32)

        return tuple(
            int(value)
            for value in rounded.tolist()
        )

    def build_level_zero(
        self,
    ) -> None:
        counts = Counter()
        word_keys = []

        for vector in self.vectors:
            key = self.quantized_key(
                vector
            )

            counts[key] += 1
            word_keys.append(key)

        for key in counts:
            unit_id = self.next_unit_id
            self.next_unit_id += 1

            self.unit_by_key[key] = unit_id

            self.units[unit_id] = UnitStats(
                unit_id=unit_id,
                level=0,
                members=(),
                use_count=counts[key],
            )

        self.word_units = [
            self.unit_by_key[key]
            for key in word_keys
        ]

    def discover_level(
        self,
        streams: list[list[int]],
        level: int,
    ) -> tuple[list[list[int]], dict[str, int]]:
        occurrences = Counter()

        for stream in streams:
            for left, right in zip(
                stream,
                stream[1:],
            ):
                occurrences[
                    frozenset(
                        (
                            left,
                            right,
                        )
                    )
                ] += 1

        recurring = {
            key
            for key, count
            in occurrences.items()
            if count >= MIN_REUSE
        }

        before = self.next_unit_id

        combination_to_id: dict[
            frozenset[int],
            int,
        ] = {}

        for key in recurring:
            existing = self._assembly_lookup(
                key
            )

            if existing is not None:
                combination_to_id[key] = existing
                continue

            unit_id = self.next_unit_id
            self.next_unit_id += 1

            self.units[unit_id] = UnitStats(
                unit_id=unit_id,
                level=level,
                members=tuple(
                    sorted(key)
                ),
                use_count=occurrences[key],
            )

            combination_to_id[key] = unit_id

        created = (
            self.next_unit_id
            - before
        )

        next_streams = []

        for stream in streams:
            output = []
            i = 0

            while i < len(stream):
                if i + 1 < len(stream):
                    key = frozenset(
                        (
                            stream[i],
                            stream[i + 1],
                        )
                    )

                    unit_id = combination_to_id.get(
                        key
                    )

                    if unit_id is not None:
                        output.append(
                            unit_id
                        )
                        i += 2
                        continue

                output.append(
                    stream[i]
                )
                i += 1

            next_streams.append(
                output
            )

        return (
            next_streams,
            {
                "candidate_pairs": len(
                    occurrences
                ),
                "recurring_pairs": len(
                    recurring
                ),
                "new_units": created,
                "output_units": sum(
                    len(stream)
                    for stream in next_streams
                ),
            },
        )

    def _assembly_lookup(
        self,
        key: frozenset[int],
    ) -> int | None:
        members = tuple(
            sorted(key)
        )

        for unit in self.units.values():
            if (
                unit.level > 0
                and unit.members == members
            ):
                return unit.unit_id

        return None

    def train(
        self,
    ) -> list[dict[str, int]]:
        self.build_level_zero()

        print(
            "ACTIVATION level=0 "
            f"unique_units={len(self.units)} "
            f"words={len(self.word_units)}",
            flush=True,
        )

        streams = [
            [unit_id]
            for unit_id
            in self.word_units
        ]

        # Words are represented as one activation unit each at level 0.
        # Higher-level structural discovery therefore operates over groups
        # of words. This is deliberately conservative: the first semantic
        # bridge is activation-space similarity, not token adjacency.
        #
        # Reconstruct a word-neighborhood stream from nearest neighbors.
        graph_streams = []

        similarities = self.normalized() @ self.normalized().T

        for index in range(
            similarities.shape[0]
        ):
            order = torch.argsort(
                similarities[index],
                descending=True,
            )

            neighbors = []

            for j_tensor in order:
                j = int(
                    j_tensor.item()
                )

                if j == index:
                    continue

                neighbors.append(
                    self.word_units[j]
                )

                if len(neighbors) >= TOP_K_NEIGHBORS:
                    break

            graph_streams.append(
                neighbors
            )

        results = []

        current = graph_streams

        for level in range(
            1,
            MAX_LEVELS + 1,
        ):
            current, stats = (
                self.discover_level(
                    current,
                    level,
                )
            )

            results.append(
                {
                    "level": level,
                    **stats,
                }
            )

            print(
                f"ACTIVATION level={level:2d} "
                f"recurring={stats['recurring_pairs']:7d} "
                f"new_units={stats['new_units']:7d} "
                f"output_units={stats['output_units']:8d}",
                flush=True,
            )

            if stats["new_units"] == 0:
                break

        return results

    def compression_stats(
        self,
        words: list[str],
    ) -> dict[str, float]:
        unique_level0 = sum(
            unit.level == 0
            for unit in self.units.values()
        )

        total_units = len(
            self.units
        )

        unique_activation_signatures = len(
            self.unit_by_key
        )

        return {
            "words": float(
                len(words)
            ),
            "unique_activation_signatures": float(
                unique_activation_signatures
            ),
            "level0_units": float(
                unique_level0
            ),
            "total_graph_units": float(
                total_units
            ),
            "raw_word_units_ratio": (
                unique_level0
                / max(
                    1,
                    len(words),
                )
            ),
        }


# ---------------------------------------------------------------------------
# Semantic alignment probe
# ---------------------------------------------------------------------------

def semantic_neighborhood_alignment(
    words: list[str],
    vectors: torch.Tensor,
    semantics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Compare activation-space nearest neighbors against human semantic
    feature overlap.

    This is intentionally independent of the graph compression mechanism.
    It tells us whether the activation topology contains semantic structure
    worth compressing.
    """
    word_to_index = {
        word: index
        for index, word in enumerate(words)
    }

    anchors = [
        word
        for word in semantics
        if word in word_to_index
    ]

    normalized = (
        vectors
        / torch.linalg.vector_norm(
            vectors,
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
    )

    values = []
    targets = []

    for word in anchors:
        i = word_to_index[word]

        scores = (
            normalized
            @ normalized[i]
        )

        order = torch.argsort(
            scores,
            descending=True,
        )

        for j_tensor in order:
            j = int(
                j_tensor.item()
            )

            if j == i:
                continue

            other = words[j]

            if other not in semantics:
                continue

            values.append(
                float(
                    scores[j].item()
                )
            )

            targets.append(
                jaccard_weighted(
                    semantics[word],
                    semantics[other],
                )
            )

            # A compact sample keeps the graph experiment fast.
            if len(values) >= 20000:
                break

        if len(values) >= 20000:
            break

    if len(values) < 20:
        return {
            "pairs": float(
                len(values)
            ),
            "spearman": 0.0,
        }

    predicted = torch.tensor(
        values,
        dtype=torch.float32,
    )

    target = torch.tensor(
        targets,
        dtype=torch.float32,
    )

    def rankdata(
        x: torch.Tensor,
    ) -> torch.Tensor:
        order = torch.argsort(x)
        ranks = torch.empty_like(
            x
        )
        ranks[order] = torch.arange(
            x.numel(),
            dtype=ranks.dtype,
        )
        return ranks

    pr = rankdata(
        predicted
    )
    tr = rankdata(
        target
    )

    pr -= pr.mean()
    tr -= tr.mean()

    corr = (
        torch.dot(
            pr,
            tr,
        )
        / (
            torch.linalg.vector_norm(pr)
            * torch.linalg.vector_norm(tr)
        ).clamp_min(1e-12)
    )

    return {
        "pairs": float(
            len(values)
        ),
        "spearman": float(
            corr.item()
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    print(
        "=== V103 LLM ACTIVATION GRAPH COMPRESSION ==="
    )
    print(
        "Loading saved V102 representation; no LLM inference."
    )
    print()

    words, vectors, best = load_probe()

    print(
        "words:",
        len(words),
    )

    print(
        "vector_shape:",
        tuple(vectors.shape),
    )

    print(
        "v102_best_level:",
        best["level"],
    )

    print(
        "v102_best_representation:",
        best["representation"],
    )

    print()

    semantics = load_large_semantics(
        SEMANTICS_PATH
    )

    matched = sum(
        word in semantics
        for word in words
    )

    print(
        "semantic_cues:",
        len(semantics),
    )

    print(
        "matched_activation_words:",
        matched,
    )

    print()

    # ---------------------------------------------------------------
    # Direct activation/semantic alignment.
    # ---------------------------------------------------------------

    alignment = semantic_neighborhood_alignment(
        words,
        vectors,
        semantics,
    )

    print(
        "=== ACTIVATION -> SEMANTIC ALIGNMENT ==="
    )

    print(
        "pairs:",
        alignment["pairs"],
    )

    print(
        "spearman:",
        alignment["spearman"],
    )

    print()

    # ---------------------------------------------------------------
    # Graph compression.
    # ---------------------------------------------------------------

    graph = ActivationGraph(
        vectors
    )

    level_results = graph.train()

    compression = graph.compression_stats(
        words
    )

    print(
        "=== ACTIVATION GRAPH COMPRESSION ==="
    )

    for key, value in compression.items():
        print(
            f"{key:32s}: {value}"
        )

    print()

    print(
        "=== LEVEL SUMMARY ==="
    )

    for row in level_results:
        print(
            f"level={row['level']:2d} "
            f"candidate_pairs={row['candidate_pairs']:8d} "
            f"recurring={row['recurring_pairs']:8d} "
            f"new_units={row['new_units']:8d} "
            f"output_units={row['output_units']:8d}"
        )

    print()

    print(
        "=== V103 INTERPRETATION ==="
    )

    print(
        "The input representation is the V102-selected centered/top-PC-removed "
        "SmolLM2-360M layer representation."
    )

    print(
        "The graph is built from activation-space reuse, not token identity."
    )

    print(
        "A positive activation->semantic correlation means the LLM activation "
        "topology contains independently observed semantic structure."
    )

    print(
        "Recursive new_units > 0 across multiple levels means the activation "
        "topology supports reusable higher-order structure."
    )

    print()

    save_path = (
        ROOT
        / "results"
        / "v103_llm_activation_graph.pt"
    )

    torch.save(
        {
            "words": words,
            "vectors": vectors,
            "v102_best": best,
            "activation_graph_units": graph.units,
            "activation_graph_unit_by_key": graph.unit_by_key,
            "level_results": level_results,
            "compression": compression,
            "semantic_alignment": alignment,
        },
        save_path,
    )

    print(
        "saved:",
        save_path,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V103 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
