from __future__ import annotations

"""
V113 — GRAPH-ONLY SEMANTIC MEMORY RECONSTRUCTION

This is the experiment described as:

    HUMAN SEMANTIC SEED
        ↓
    frozen LLM descriptions
        ↓
    V111 graph memory
        ↓
    forget the original LLM descriptions
        ↓
    query the graph alone

The V111 JSON contains the learned word -> concept links. We deliberately
do NOT use:
    * SmolLM2
    * semantics-large.csv for candidate construction
    * the original LLM outputs

For reconstruction we use only:
    * the learned graph's word -> concept edges
    * graph concept usage / co-occurrence
    * dictionary lexical relationships

Evaluation uses semantics-large.csv ONLY AFTER prediction, as an external
gold standard.

Two tests are reported:

1. MEMORY RECONSTRUCTION
   Mask each target word's own learned concept edges, then reconstruct its
   concept set from graph-connected lexical neighbors.

   This answers:
       Can the persistent graph regenerate a forgotten word representation?

2. HUMAN SEMANTIC RECONSTRUCTION
   Compare the reconstructed graph concepts against the human semantic gold
   vocabulary.

Comparisons:
    DIRECT GRAPH LOOKUP
        trivial upper-bound / stored memory

    GRAPH-ONLY HELD-OUT
        target's own links removed

    LEXICAL-ONLY GOLD BASELINE
        human semantic features retrieved from lexical neighbors

The important distinction:
    the seed semantic corpus is available only as evaluation ground truth in
    the final scoring. The graph reconstruction itself has no access to it.

This is intentionally a graph-memory experiment, not another LLM experiment.
"""

import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MEMORY_PATH = (
    ROOT
    / "results"
    / "v111_compact_semantic_memory.json"
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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v113_graph_only_reconstruction.json"
)

# 0.20 means 20% of words are evaluated in the masked reconstruction test.
HOLDOUT_FRACTION = 0.20

SEED = 9173

# Number of lexical neighbors used to infer a masked word.
LEXICAL_NEIGHBORS = 12

# Number of graph concepts returned.
MAX_CONCEPTS = 8

# Co-usage neighborhood.
TOP_CONCEPTS_FROM_NEIGHBORS = 16


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def load_dictionary(
    path: Path,
) -> list[str]:
    words = set()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw in handle:
            word = raw.strip().lower()

            if word and word.isalpha():
                words.add(word)

    return sorted(words)


# ---------------------------------------------------------------------------
# Human semantic gold
# ---------------------------------------------------------------------------

class HumanSemanticGold:
    def __init__(self) -> None:
        self.cue_features: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

    def load(
        self,
        path: Path,
    ) -> None:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                cue = row.get(
                    "cue",
                    "",
                ).strip().lower()

                feature = row.get(
                    "translated",
                    "",
                ).strip().lower()

                if not cue or not feature:
                    continue

                try:
                    weight = float(
                        row.get(
                            "normalized_translated",
                            0.0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    weight = 0.0

                if weight <= 0:
                    try:
                        frequency = float(
                            row.get(
                                "frequency_translated",
                                0.0,
                            )
                        )

                        n = float(
                            row.get(
                                "n",
                                0.0,
                            )
                        )

                        if n > 0:
                            weight = frequency / n
                    except (
                        TypeError,
                        ValueError,
                    ):
                        weight = 0.0

                if weight > 0:
                    self.cue_features[
                        cue
                    ][feature] += weight

    def gold(
        self,
        word: str,
        limit: int = MAX_CONCEPTS,
    ) -> set[str]:
        return set(
            feature
            for feature, _weight
            in self.cue_features.get(
                word,
                Counter(),
            ).most_common(
                limit
            )
        )


# ---------------------------------------------------------------------------
# V111 graph memory
# ---------------------------------------------------------------------------

class GraphMemory:
    def __init__(
        self,
        path: Path,
    ) -> None:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.concept_id_by_name = {
            str(name): int(identifier)
            for name, identifier
            in payload[
                "concept_id_by_name"
            ].items()
        }

        self.concept_name_by_id = {
            identifier: name
            for name, identifier
            in self.concept_id_by_name.items()
        }

        self.usage = Counter(
            {
                int(identifier): int(count)
                for identifier, count
                in payload.get(
                    "usage",
                    {},
                ).items()
            }
        )

        self.word_concepts: dict[
            str,
            list[int],
        ] = {
            str(word): [
                int(identifier)
                for identifier
                in identifiers
            ]
            for word, identifiers
            in payload.get(
                "word_concepts",
                {},
            ).items()
        }

        self.words_by_concept: dict[
            int,
            set[str],
        ] = defaultdict(set)

        # Reconstruct concept -> words from the saved word -> concept graph.
        for word, identifiers in self.word_concepts.items():
            for identifier in identifiers:
                self.words_by_concept[
                    identifier
                ].add(word)

        # Co-usage graph: concepts that repeatedly occur together on words.
        self.co_usage: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        for identifiers in self.word_concepts.values():
            unique = list(
                dict.fromkeys(
                    identifiers
                )
            )

            for i, left in enumerate(unique):
                for right in unique[
                    i + 1:
                ]:
                    self.co_usage[
                        left
                    ][right] += 1
                    self.co_usage[
                        right
                    ][left] += 1

    def concept_names(
        self,
        identifiers: list[int],
    ) -> list[str]:
        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier in identifiers
            if identifier in self.concept_name_by_id
        ]

    def direct(
        self,
        word: str,
    ) -> list[str]:
        return self.concept_names(
            self.word_concepts.get(
                word,
                [],
            )
        )[:MAX_CONCEPTS]

    def neighbor_words(
        self,
        word: str,
        dictionary_set: set[str],
    ) -> list[str]:
        """
        Fast lexical neighborhood:
            same 3-char prefix
            same 3-char suffix
            closest length

        No semantic corpus is used.
        """
        prefix = word[:3]
        suffix = word[-3:]

        candidates = []

        for other in dictionary_set:
            if other == word:
                continue

            score = 0

            if other.startswith(prefix):
                score += 4

            if other.endswith(suffix):
                score += 4

            if abs(
                len(other)
                - len(word)
            ) <= 1:
                score += 1

            if score > 0:
                candidates.append(
                    (
                        score,
                        other,
                    )
                )

        candidates.sort(
            key=lambda item: (
                -item[0],
                abs(
                    len(item[1])
                    - len(word)
                ),
                item[1],
            )
        )

        return [
            other
            for _score, other
            in candidates[
                :LEXICAL_NEIGHBORS
            ]
        ]

    def reconstruct(
        self,
        word: str,
        dictionary_set: set[str],
        masked: bool = True,
    ) -> list[str]:
        """
        Reconstruct a word representation from graph structure.

        The target's own links are ignored when masked=True.

        Score sources:
            * concepts used by lexical neighbors
            * frequency of concept use among neighbors
            * concept co-usage structure
        """
        if not masked:
            direct = self.direct(
                word
            )

            if direct:
                return direct

        neighbors = self.neighbor_words(
            word,
            dictionary_set,
        )

        scores = Counter()

        # Concept evidence from neighboring words.
        neighbor_concepts = []

        for neighbor in neighbors:
            identifiers = (
                self.word_concepts.get(
                    neighbor,
                    [],
                )
            )

            local = set(
                identifiers
            )

            for identifier in local:
                scores[
                    identifier
                ] += 1.0

            neighbor_concepts.append(
                local
            )

        # Concepts that recur across several neighbors get a bonus.
        neighbor_frequency = Counter()

        for concepts in neighbor_concepts:
            for identifier in concepts:
                neighbor_frequency[
                    identifier
                ] += 1

        for identifier, count in (
            neighbor_frequency.items()
        ):
            if count >= 2:
                scores[
                    identifier
                ] += (
                    2.0
                    * count
                )

        # Concept co-usage provides a second graph-only signal.
        top_neighbor_concepts = [
            identifier
            for identifier, _score
            in scores.most_common(
                TOP_CONCEPTS_FROM_NEIGHBORS
            )
        ]

        for identifier in top_neighbor_concepts:
            for related, count in (
                self.co_usage.get(
                    identifier,
                    Counter(),
                ).most_common(8)
            ):
                scores[
                    related
                ] += (
                    0.25
                    * count
                )

        # Do not allow the target's own concept nodes to leak into a masked
        # reconstruction merely because of co-usage.
        target_ids = set(
            self.word_concepts.get(
                word,
                [],
            )
        )

        if masked:
            for identifier in target_ids:
                scores.pop(
                    identifier,
                    None,
                )

        ranked = [
            identifier
            for identifier, _score
            in scores.most_common(
                MAX_CONCEPTS
            )
        ]

        return self.concept_names(
            ranked
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_score(
    predicted: set[str],
    gold: set[str],
) -> float:
    if not predicted or not gold:
        return 0.0

    hits = len(
        predicted & gold
    )

    precision = (
        hits
        / len(predicted)
    )

    recall = (
        hits
        / len(gold)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def score_condition(
    predictions: dict[str, list[str]],
    gold: HumanSemanticGold,
) -> dict[str, float]:
    precision = []
    recall = []
    f1 = []
    predicted_count = []

    evaluated = 0

    for word, concepts in predictions.items():
        target = gold.gold(
            word
        )

        if not target:
            continue

        predicted = set(
            concepts
        )

        hits = len(
            predicted
            & target
        )

        precision.append(
            hits
            / max(
                1,
                len(predicted),
            )
        )

        recall.append(
            hits
            / max(
                1,
                len(target),
            )
        )

        f1.append(
            f1_score(
                predicted,
                target,
            )
        )

        predicted_count.append(
            len(predicted)
        )

        evaluated += 1

    return {
        "evaluated_words": float(
            evaluated
        ),
        "mean_precision": (
            sum(precision)
            / max(
                1,
                len(precision),
            )
        ),
        "mean_recall": (
            sum(recall)
            / max(
                1,
                len(recall),
            )
        ),
        "mean_f1": (
            sum(f1)
            / max(
                1,
                len(f1),
            )
        ),
        "mean_predicted_concepts": (
            sum(predicted_count)
            / max(
                1,
                len(predicted_count),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V113 GRAPH-ONLY SEMANTIC MEMORY RECONSTRUCTION ==="
    )

    if not MEMORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing V111 memory: {MEMORY_PATH}"
        )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    dictionary_set = set(
        words
    )

    print(
        "dictionary_words:",
        len(words),
    )

    graph = GraphMemory(
        MEMORY_PATH
    )

    print(
        "graph_concepts:",
        len(
            graph.concept_id_by_name
        ),
    )

    print(
        "graph_words:",
        len(
            graph.word_concepts
        ),
    )

    gold = HumanSemanticGold()
    gold.load(
        SEMANTICS_PATH
    )

    # ---------------------------------------------------------------
    # Deterministic holdout set.
    # ---------------------------------------------------------------

    rng = random.Random(
        SEED
    )

    eligible = [
        word
        for word in words
        if word in graph.word_concepts
        and gold.gold(word)
    ]

    rng.shuffle(
        eligible
    )

    holdout_count = max(
        1,
        int(
            len(eligible)
            * HOLDOUT_FRACTION
        ),
    )

    holdout = sorted(
        eligible[
            :holdout_count
        ]
    )

    print(
        "held_out_words:",
        len(holdout),
    )

    print()

    # ---------------------------------------------------------------
    # A. Direct stored memory — sanity / upper bound.
    # ---------------------------------------------------------------

    direct_predictions = {}

    for word in holdout:
        direct_predictions[
            word
        ] = graph.direct(
            word
        )

    direct_scores = score_condition(
        direct_predictions,
        gold,
    )

    print(
        "=== A DIRECT GRAPH LOOKUP ==="
    )

    for key, value in direct_scores.items():
        print(
            f"{key}: {value}"
        )

    print()

    # ---------------------------------------------------------------
    # B. Masked graph-only reconstruction.
    # ---------------------------------------------------------------

    masked_predictions = {}

    reconstruction_started = time.perf_counter()

    for index, word in enumerate(
        holdout,
        start=1,
    ):
        masked_predictions[
            word
        ] = graph.reconstruct(
            word,
            dictionary_set,
            masked=True,
        )

        if (
            index <= 10
            or index % 100 == 0
            or index == len(holdout)
        ):
            print(
                f"RECONSTRUCT "
                f"{index:4d}/{len(holdout):4d} "
                f"{word:20s} -> "
                f"{masked_predictions[word]}",
                flush=True,
            )

    masked_scores = score_condition(
        masked_predictions,
        gold,
    )

    print()

    print(
        "=== B GRAPH-ONLY MASKED RECONSTRUCTION ==="
    )

    for key, value in masked_scores.items():
        print(
            f"{key}: {value}"
        )

    print(
        "reconstruction_seconds:",
        f"{time.perf_counter() - reconstruction_started:.3f}",
    )

    print()

    # ---------------------------------------------------------------
    # C. Compare graph reconstruction against the word's learned V111
    # representation itself. This is NOT the human gold; it measures whether
    # the graph can regenerate the concepts it previously stored when those
    # direct edges are unavailable.
    # ---------------------------------------------------------------

    internal_f1 = []

    for word in holdout:
        learned = set(
            graph.direct(
                word
            )
        )

        reconstructed = set(
            masked_predictions.get(
                word,
                [],
            )
        )

        internal_f1.append(
            f1_score(
                reconstructed,
                learned,
            )
        )

    mean_internal = (
        sum(internal_f1)
        / max(
            1,
            len(internal_f1),
        )
    )

    print(
        "=== C RECONSTRUCTED vs ORIGINAL LEARNED MEMORY ==="
    )

    print(
        "mean_f1:",
        mean_internal,
    )

    # ---------------------------------------------------------------
    # D. Example cases.
    # ---------------------------------------------------------------

    print()

    print(
        "=== EXAMPLES ==="
    )

    shown = 0

    for word in holdout:
        original = graph.direct(
            word
        )

        reconstructed = (
            masked_predictions[word]
        )

        target = sorted(
            gold.gold(
                word
            )
        )

        # Prefer examples where reconstruction is non-empty.
        if not reconstructed:
            continue

        print(
            f"{word:20s}"
        )

        print(
            "  original_graph:",
            original,
        )

        print(
            "  reconstructed :",
            reconstructed,
        )

        print(
            "  human_gold    :",
            target,
        )

        shown += 1

        if shown >= 20:
            break

    # ---------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------

    payload = {
        "experiment": "V113 graph-only semantic memory reconstruction",
        "holdout_fraction": HOLDOUT_FRACTION,
        "seed": SEED,
        "held_out_words": holdout,
        "direct_scores": direct_scores,
        "graph_only_masked_scores": masked_scores,
        "reconstructed_vs_original_learned_f1": mean_internal,
        "predictions": {
            word: {
                "direct": graph.direct(word),
                "reconstructed": masked_predictions[word],
                "human_gold": sorted(
                    gold.gold(word)
                ),
            }
            for word in holdout
        },
        "graph_concepts": len(
            graph.concept_id_by_name
        ),
        "graph_words": len(
            graph.word_concepts
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "=== V113 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
