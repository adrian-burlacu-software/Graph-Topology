from __future__ import annotations

"""
V121 — CONCEPT TOPOLOGY INDUCTION

Purpose
-------
V120 showed that the V111 graph has a compressed vocabulary, but the graph's
candidate neighborhoods are noisy. The next question is whether the
word->concept bipartite memory contains enough information to induce a useful
CONCEPT->CONCEPT topology.

This experiment does NOT use the LLM.

It loads:
    results/v111_compact_semantic_memory.json

and constructs a concept graph from the learned word->concept assignments.

For each pair of concepts A/B we compute:

    co_occurrence(A,B)
    P(B|A)
    P(A|B)
    Jaccard(A,B)
    PMI(A,B)

We then derive three interpretable edge classes:

    PEER
        symmetric co-occurrence / similarity

    IMPLICATION_CANDIDATE
        strongly asymmetric conditional probability

    ASSOCIATION
        positive co-occurrence without strong asymmetry

These are NOT claimed to be human semantic relation labels. They are
topological hypotheses generated solely from the learned graph.

Main evaluation
---------------
Mask part of a word's learned concept representation.

Given the remaining concept cues, retrieve missing concepts from the induced
concept graph.

Compare:

    DIRECT
        stored V111 representation

    TOPOLOGY
        induced concept graph

    LEXICAL
        concepts attached to lexical neighbors

against:

    A) V111 learned representation
    B) human semantic gold (semantics-large.csv)

This directly tests whether the graph has learned reusable RELATIONS among
concepts, rather than merely memorizing word->concept edges.

No graph mutation.
No LLM.
No semantic corpus during topology induction.
The human semantic corpus is used only for final evaluation.
"""

import csv
import json
import math
import random
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
    / "v121_concept_topology_induction.json"
)

SEED = 12021

# Fraction of the learned word representation to hide.
MASK_FRACTION = 0.50

# Minimum word degree for meaningful masking.
MIN_WORD_CONCEPTS = 4

# Number of topology neighbors considered per concept.
CONCEPT_NEIGHBORS = 32

# Number of reconstructed concepts returned.
MAX_PREDICTIONS = 8

# Evaluation set.
EVAL_WORDS = 1500


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


def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    prefix = defaultdict(list)
    suffix = defaultdict(list)

    for word in words:
        prefix[word[:3]].append(word)
        suffix[word[-3:]].append(word)

    result = {}

    for word in words:
        candidates = set(
            prefix[word[:3]]
        )
        candidates.update(
            suffix[word[-3:]]
        )
        candidates.discard(word)

        ranked = sorted(
            candidates,
            key=lambda other: (
                not other.startswith(word[:3]),
                not other.endswith(word[-3:]),
                abs(len(other) - len(word)),
                other,
            ),
        )

        result[word] = ranked[:12]

    return result


# ---------------------------------------------------------------------------
# Human gold
# ---------------------------------------------------------------------------

class HumanGold:
    def __init__(self) -> None:
        self.cue_features = defaultdict(Counter)

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
        limit: int = 8,
    ) -> set[str]:
        return {
            feature
            for feature, _weight
            in self.cue_features.get(
                word,
                Counter(),
            ).most_common(limit)
        }


# ---------------------------------------------------------------------------
# Learned bipartite memory
# ---------------------------------------------------------------------------

class LearnedGraph:
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

        self.word_concepts = {
            str(word): list(
                map(
                    int,
                    ids,
                )
            )
            for word, ids
            in payload[
                "word_concepts"
            ].items()
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

        self.words_by_concept = defaultdict(set)

        for word, identifiers in self.word_concepts.items():
            for identifier in identifiers:
                self.words_by_concept[
                    identifier
                ].add(word)

        # Undirected concept-pair co-occurrence.
        self.co_occurrence = defaultdict(Counter)

        for identifiers in self.word_concepts.values():
            unique = list(
                dict.fromkeys(
                    identifiers
                )
            )

            for i, left in enumerate(unique):
                for right in unique[i + 1:]:
                    self.co_occurrence[
                        left
                    ][right] += 1

                    self.co_occurrence[
                        right
                    ][left] += 1

        self.word_count = len(
            self.word_concepts
        )

        self.concept_document_frequency = Counter(
            {
                identifier: len(words)
                for identifier, words
                in self.words_by_concept.items()
            }
        )

        self.typed_edges = defaultdict(dict)


    def concepts_for_word(
        self,
        word: str,
    ) -> list[str]:
        return [
            self.concept_name_by_id[identifier]
            for identifier
            in self.word_concepts.get(
                word,
                [],
            )
            if identifier in self.concept_name_by_id
        ]


    # -----------------------------------------------------------------------
    # Topology construction
    # -----------------------------------------------------------------------

    def conditional_probability(
        self,
        source: int,
        target: int,
    ) -> float:
        co = self.co_occurrence[
            source
        ][target]

        denominator = self.concept_document_frequency[
            source
        ]

        if denominator <= 0:
            return 0.0

        return co / denominator

    def jaccard(
        self,
        a: int,
        b: int,
    ) -> float:
        wa = self.words_by_concept[a]
        wb = self.words_by_concept[b]

        union = len(
            wa | wb
        )

        if union == 0:
            return 0.0

        return len(
            wa & wb
        ) / union

    def pmi(
        self,
        a: int,
        b: int,
    ) -> float:
        co = self.co_occurrence[a][b]

        if co <= 0:
            return float("-inf")

        total = max(
            1,
            self.word_count,
        )

        pa = (
            self.concept_document_frequency[a]
            / total
        )

        pb = (
            self.concept_document_frequency[b]
            / total
        )

        pab = co / total

        if pa <= 0 or pb <= 0 or pab <= 0:
            return float("-inf")

        return math.log(
            pab / (
                pa * pb
            )
        )

    def build_typed_topology(
        self,
    ) -> None:
        """
        Keep only a sparse topological graph.

        Edge class heuristic:
            PEER
                roughly symmetric usage and strong positive association

            IMPLICATION_CANDIDATE
                P(target|source) much larger than P(source|target)

            ASSOCIATION
                everything else sufficiently supported
        """
        for source, neighbors in self.co_occurrence.items():
            scored = []

            for target, co in neighbors.items():
                if co < 2:
                    continue

                p_target_given_source = (
                    self.conditional_probability(
                        source,
                        target,
                    )
                )

                p_source_given_target = (
                    self.conditional_probability(
                        target,
                        source,
                    )
                )

                jaccard = self.jaccard(
                    source,
                    target,
                )

                pmi = self.pmi(
                    source,
                    target,
                )

                if not math.isfinite(pmi):
                    continue

                ratio = (
                    p_target_given_source
                    / max(
                        p_source_given_target,
                        1e-9,
                    )
                )

                if (
                    ratio >= 3.0
                    and p_target_given_source >= 0.20
                ):
                    relation = (
                        "IMPLICATION_CANDIDATE"
                    )
                    direction_score = ratio

                elif (
                    jaccard >= 0.10
                    and pmi > 0
                ):
                    relation = "PEER"
                    direction_score = 1.0

                elif pmi > 0:
                    relation = "ASSOCIATION"
                    direction_score = 1.0

                else:
                    continue

                # Composite topology strength.
                strength = (
                    math.log1p(co)
                    * (
                        1.0
                        + max(
                            0.0,
                            pmi,
                        )
                    )
                    * (
                        0.5
                        + jaccard
                    )
                )

                scored.append(
                    (
                        strength,
                        target,
                        relation,
                        p_target_given_source,
                        p_source_given_target,
                        jaccard,
                        pmi,
                        direction_score,
                        co,
                    )
                )

            scored.sort(
                key=lambda item: (
                    -item[0],
                    self.concept_name_by_id[
                        item[1]
                    ],
                )
            )

            for row in scored[
                :CONCEPT_NEIGHBORS
            ]:
                (
                    strength,
                    target,
                    relation,
                    p_t_s,
                    p_s_t,
                    jaccard,
                    pmi,
                    direction_score,
                    co,
                ) = row

                self.typed_edges[
                    source
                ][
                    target
                ] = {
                    "relation": relation,
                    "strength": strength,
                    "p_target_given_source": p_t_s,
                    "p_source_given_target": p_s_t,
                    "jaccard": jaccard,
                    "pmi": pmi,
                    "direction_score": direction_score,
                    "co_occurrence": co,
                }

    # -----------------------------------------------------------------------
    # Topology retrieval
    # -----------------------------------------------------------------------

    def retrieve_from_cues(
        self,
        cue_ids: list[int],
        excluded: set[int],
    ) -> list[tuple[int, float]]:
        """
        Multi-hop concept retrieval.

        Direct neighbors get more weight.
        Typed PEER and ASSOCIATION edges contribute normally.
        IMPLICATION_CANDIDATE edges are slightly directional.

        Returns ranked candidate concept IDs.
        """
        scores = Counter()

        for cue in cue_ids:
            for target, edge in self.typed_edges.get(
                cue,
                {},
            ).items():
                if target in excluded:
                    continue

                relation = edge["relation"]
                strength = edge["strength"]

                if relation == "PEER":
                    weight = 1.0
                elif relation == "ASSOCIATION":
                    weight = 0.75
                else:
                    weight = 1.15

                scores[
                    target
                ] += (
                    weight
                    * strength
                )

        # Second hop through the strongest first-hop concepts.
        first_hop = [
            identifier
            for identifier, _score
            in scores.most_common(16)
        ]

        for intermediary in first_hop:
            for target, edge in self.typed_edges.get(
                intermediary,
                {},
            ).items():
                if target in excluded:
                    continue

                scores[
                    target
                ] += (
                    0.20
                    * edge["strength"]
                )

        return scores.most_common(
            MAX_PREDICTIONS
        )

    # -----------------------------------------------------------------------
    # Lexical baseline from the learned graph
    # -----------------------------------------------------------------------

    def lexical_reconstruct(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[int]:
        scores = Counter()

        for neighbor in lexical_index.get(
            word,
            [],
        ):
            for identifier in self.word_concepts.get(
                neighbor,
                [],
            ):
                scores[
                    identifier
                ] += 1

        return [
            identifier
            for identifier, _score
            in scores.most_common(
                MAX_PREDICTIONS
            )
        ]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: set[str],
    gold: set[str],
) -> tuple[float, float, float]:
    if not predicted or not gold:
        return 0.0, 0.0, 0.0

    hits = len(
        predicted & gold
    )

    precision = hits / len(
        predicted
    )

    recall = hits / len(
        gold
    )

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

    return precision, recall, f1


def evaluate_condition(
    predictions: dict[str, list[str]],
    gold_sets: dict[str, set[str]],
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []

    for word, predicted in predictions.items():
        gold = gold_sets.get(
            word,
            set(),
        )

        if not gold:
            continue

        p, r, f = precision_recall_f1(
            set(predicted),
            gold,
        )

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "evaluated": len(f1s),
        "precision": (
            sum(precisions)
            / max(
                1,
                len(precisions),
            )
        ),
        "recall": (
            sum(recalls)
            / max(
                1,
                len(recalls),
            )
        ),
        "f1": (
            sum(f1s)
            / max(
                1,
                len(f1s),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V121 CONCEPT TOPOLOGY INDUCTION ==="
    )

    graph = LearnedGraph(
        MEMORY_PATH
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    lexical_index = build_lexical_index(
        words
    )

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    print(
        "words_in_graph:",
        graph.word_count,
    )

    print(
        "concepts:",
        len(
            graph.concept_id_by_name
        ),
    )

    # ---------------------------------------------------------------
    # Learn concept topology.
    # ---------------------------------------------------------------

    topology_started = time.perf_counter()

    graph.build_typed_topology()

    edge_count = sum(
        len(targets)
        for targets in graph.typed_edges.values()
    )

    relation_counts = Counter()

    for targets in graph.typed_edges.values():
        for edge in targets.values():
            relation_counts[
                edge["relation"]
            ] += 1

    print(
        "typed_edges:",
        edge_count,
    )

    print(
        "relation_counts:",
        dict(relation_counts),
    )

    print(
        "topology_seconds:",
        f"{time.perf_counter() - topology_started:.3f}",
    )

    # ---------------------------------------------------------------
    # Choose words with enough learned concepts to mask.
    # ---------------------------------------------------------------

    candidates = [
        word
        for word in words
        if len(
            graph.word_concepts.get(
                word,
                [],
            )
        ) >= MIN_WORD_CONCEPTS
    ]

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        candidates
    )

    evaluation_words = sorted(
        candidates[
            :EVAL_WORDS
        ]
    )

    print(
        "evaluation_words:",
        len(evaluation_words),
    )

    # Gold set.
    gold_sets = {
        word: gold.gold(
            word
        )
        for word in evaluation_words
        if gold.gold(word)
    }

    # ---------------------------------------------------------------
    # Mask each target word.
    # ---------------------------------------------------------------

    topology_predictions = {}
    lexical_predictions = {}
    direct_predictions = {}

    topology_internal_predictions = {}
    lexical_internal_predictions = {}

    for index, word in enumerate(
        evaluation_words,
        start=1,
    ):
        original_ids = list(
            graph.word_concepts[word]
        )

        if len(original_ids) < MIN_WORD_CONCEPTS:
            continue

        mask_count = max(
            1,
            int(
                len(original_ids)
                * MASK_FRACTION
            ),
        )

        # Random but deterministic hidden set.
        rng_word = random.Random(
            hash(
                (
                    SEED,
                    word,
                )
            )
            & 0xFFFFFFFF
        )

        shuffled = list(
            original_ids
        )

        rng_word.shuffle(
            shuffled
        )

        hidden = set(
            shuffled[
                :mask_count
            ]
        )

        visible = [
            identifier
            for identifier
            in original_ids
            if identifier not in hidden
        ]

        # DIRECT is the hidden part itself — this is the ceiling for the
        # "how much semantic information was there to recover?" question.
        direct_hidden = [
            graph.concept_name_by_id[
                identifier
            ]
            for identifier in hidden
        ]

        direct_predictions[
            word
        ] = direct_hidden

        # Topology gets only visible concept cues.
        topology_ranked = graph.retrieve_from_cues(
            visible,
            hidden | set(visible),
        )

        topology_predictions[
            word
        ] = [
            graph.concept_name_by_id[
                identifier
            ]
            for identifier, _score
            in topology_ranked
        ]

        # Lexical baseline is independent of hidden target edges.
        lexical_ids = graph.lexical_reconstruct(
            word,
            lexical_index,
        )

        # Do not let lexical baseline accidentally return masked target
        # concepts as a direct target lookup.
        lexical_predictions[
            word
        ] = [
            graph.concept_name_by_id[
                identifier
            ]
            for identifier in lexical_ids
            if identifier not in hidden
        ][:MAX_PREDICTIONS]

        # Internal learned-graph target: evaluate recovery against the hidden
        # concepts, not against human gold.
        topology_internal_predictions[
            word
        ] = topology_predictions[
            word
        ]

        lexical_internal_predictions[
            word
        ] = lexical_predictions[
            word
        ]

        if (
            index <= 10
            or index % 250 == 0
            or index == len(evaluation_words)
        ):
            print(
                f"EVAL "
                f"{index:4d}/{len(evaluation_words):4d} "
                f"{word:20s} "
                f"visible={len(visible)} "
                f"hidden={len(hidden)}",
                flush=True,
            )

    # ---------------------------------------------------------------
    # Evaluate against original learned memory.
    # ---------------------------------------------------------------

    hidden_gold_sets = {}

    for word in topology_predictions:
        original = list(
            graph.word_concepts[word]
        )

        mask_count = max(
            1,
            int(
                len(original)
                * MASK_FRACTION
            ),
        )

        rng_word = random.Random(
            hash(
                (
                    SEED,
                    word,
                )
            )
            & 0xFFFFFFFF
        )

        shuffled = list(
            original
        )

        rng_word.shuffle(
            shuffled
        )

        hidden = set(
            shuffled[
                :mask_count
            ]
        )

        hidden_gold_sets[
            word
        ] = {
            graph.concept_name_by_id[
                identifier
            ]
            for identifier in hidden
        }

    topology_recovery = evaluate_condition(
        topology_internal_predictions,
        hidden_gold_sets,
    )

    lexical_recovery = evaluate_condition(
        lexical_internal_predictions,
        hidden_gold_sets,
    )

    # ---------------------------------------------------------------
    # Evaluate against human semantics.
    # ---------------------------------------------------------------

    topology_human = evaluate_condition(
        topology_predictions,
        gold_sets,
    )

    lexical_human = evaluate_condition(
        lexical_predictions,
        gold_sets,
    )

    # ---------------------------------------------------------------
    # Direct learned memory vs human gold.
    # ---------------------------------------------------------------

    direct_human = evaluate_condition(
        {
            word: graph.concepts_for_word(
                word
            )
            for word in evaluation_words
        },
        gold_sets,
    )

    # ---------------------------------------------------------------
    # Save useful examples.
    # ---------------------------------------------------------------

    examples = []

    for word in evaluation_words:
        if word not in topology_predictions:
            continue

        examples.append(
            {
                "word": word,
                "original": graph.concepts_for_word(
                    word
                )[:16],
                "topology_reconstruction": topology_predictions[
                    word
                ],
                "lexical_reconstruction": lexical_predictions[
                    word
                ],
                "human_gold": sorted(
                    gold_sets.get(
                        word,
                        set(),
                    )
                ),
            }
        )

        if len(examples) >= 50:
            break

    print()
    print(
        "=== V121 INTERNAL GRAPH RECOVERY ==="
    )

    print(
        "TOPOLOGY:",
        topology_recovery,
    )

    print(
        "LEXICAL:",
        lexical_recovery,
    )

    print()
    print(
        "=== V121 HUMAN SEMANTIC GOLD ==="
    )

    print(
        "TOPOLOGY:",
        topology_human,
    )

    print(
        "LEXICAL:",
        lexical_human,
    )

    print(
        "DIRECT V111:",
        direct_human,
    )

    print()
    print(
        "topology_minus_lexical_internal_f1:",
        topology_recovery["f1"]
        - lexical_recovery["f1"],
    )

    print(
        "topology_minus_lexical_human_f1:",
        topology_human["f1"]
        - lexical_human["f1"],
    )

    print()
    print(
        "=== EXAMPLES ==="
    )

    for item in examples:
        print(
            item
        )

    payload = {
        "experiment": (
            "V121 concept topology induction"
        ),
        "seed": SEED,
        "mask_fraction": MASK_FRACTION,
        "evaluation_words": evaluation_words,
        "concept_count": len(
            graph.concept_id_by_name
        ),
        "typed_edge_count": edge_count,
        "relation_counts": dict(
            relation_counts
        ),
        "topology_internal_recovery": topology_recovery,
        "lexical_internal_recovery": lexical_recovery,
        "topology_human": topology_human,
        "lexical_human": lexical_human,
        "direct_v111_human": direct_human,
        "topology_minus_lexical_internal_f1": (
            topology_recovery["f1"]
            - lexical_recovery["f1"]
        ),
        "topology_minus_lexical_human_f1": (
            topology_human["f1"]
            - lexical_human["f1"]
        ),
        "examples": examples,
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
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V121 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
