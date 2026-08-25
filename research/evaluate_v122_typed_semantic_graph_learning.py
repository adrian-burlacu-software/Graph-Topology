from __future__ import annotations

"""
V122 — TYPED SEMANTIC GRAPH LEARNING

This rebuilds the semantic memory around typed relations instead of the flat:

    word -> concept

representation that V111 used.

Core representation:

    (WORD, RELATION, CONCEPT)

Relations:
    CATEGORY
    PROPERTY
    ACTION
    USE
    PART
    RELATION

The frozen SmolLM2 model is NOT asked to invent semantic strings.

For each word/relation:

    word
      ↓
    retrieve ~32 candidate concepts from the human semantic seed
      ↓
    LLM selects candidate NUMBERS only
      ↓
    graph stores typed edges:
        word --RELATION--> concept

The human semantic corpus is a SEED VOCABULARY / candidate source and
evaluation source. It does not provide the answer directly.

This is deliberately much closer to the intended architecture:

    lexical units
          +
    semantic seed vocabulary
          ↓
    frozen LLM teacher
          ↓
    typed semantic graph
          ↓
    reusable relation structure

No free-form concept generation.
No autonomous question generation.
No graph mutation of V111.
No .pt activation file.

The output is a new typed graph that can be queried later.
"""

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "llm"
    / "SmolLM2-360M-Instruct"
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

OUTPUT_GRAPH = (
    ROOT
    / "results"
    / "v122_typed_semantic_graph.json"
)

OUTPUT_REPORT = (
    ROOT
    / "results"
    / "v122_typed_semantic_graph_report.json"
)

# Full dictionary.
MAX_WORDS = None

BATCH_SIZE = 128

RELATIONS = (
    "CATEGORY",
    "PROPERTY",
    "ACTION",
    "USE",
    "PART",
    "RELATION",
)

CANDIDATE_LIMIT = 32

MAX_SELECTED = 4

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 12

PRINT_EVERY = 128

# Candidate retrieval:
# direct semantic-corpus cue matches get a strong boost;
# lexical-neighbor cue matches are a secondary source;
# global seed concepts are a weak fallback.
LEXICAL_NEIGHBORS = 12
GLOBAL_FALLBACK = 16

TRACE_WORDS = {
    "hello",
    "greeting",
    "dog",
    "animal",
    "ability",
    "abandon",
    "water",
    "music",
    "chair",
    "car",
}


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

    result = sorted(words)

    if MAX_WORDS is not None:
        result = result[:MAX_WORDS]

    if not result:
        raise RuntimeError(
            "No dictionary words found."
        )

    return result


def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    prefix: dict[str, list[str]] = defaultdict(list)
    suffix: dict[str, list[str]] = defaultdict(list)

    for word in words:
        prefix[word[:3]].append(word)
        suffix[word[-3:]].append(word)

    result: dict[str, list[str]] = {}

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
                abs(
                    len(other)
                    - len(word)
                ),
                other,
            ),
        )

        result[word] = ranked[
            :LEXICAL_NEIGHBORS
        ]

    return result


# ---------------------------------------------------------------------------
# Human semantic seed / gold
# ---------------------------------------------------------------------------

class SemanticSeed:
    def __init__(self) -> None:
        self.cue_features: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        self.feature_weight: Counter[str] = Counter()

    def load(
        self,
        path: Path,
    ) -> None:
        started = time.perf_counter()

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle)

            required = {
                "cue",
                "translated",
                "frequency_translated",
                "normalized_translated",
                "n",
            }

            missing = required - set(
                reader.fieldnames or []
            )

            if missing:
                raise RuntimeError(
                    "semantics-large.csv missing: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

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

                if weight <= 0.0:
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
                            weight = (
                                frequency / n
                            )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        weight = 0.0

                if weight <= 0.0:
                    continue

                self.cue_features[
                    cue
                ][feature] += weight

                self.feature_weight[
                    feature
                ] += weight

        print(
            "semantic_cues:",
            len(self.cue_features),
            flush=True,
        )

        print(
            "semantic_concepts:",
            len(self.feature_weight),
            flush=True,
        )

        print(
            "semantic_load_seconds:",
            f"{time.perf_counter() - started:.3f}",
            flush=True,
        )

    def features_for_word(
        self,
        word: str,
    ) -> list[tuple[str, float]]:
        return list(
            self.cue_features.get(
                word,
                Counter(),
            ).items()
        )


# ---------------------------------------------------------------------------
# Typed graph
# ---------------------------------------------------------------------------

class TypedGraph:
    def __init__(self) -> None:
        self.concept_id_by_name: dict[
            str,
            int,
        ] = {}

        self.concept_name_by_id: dict[
            int,
            str,
        ] = {}

        self.edges: dict[
            str,
            dict[str, list[int]],
        ] = defaultdict(
            lambda: defaultdict(list)
        )

        self.edge_count = 0

        self.selection_usage: Counter[
            str
        ] = Counter()

    def get_or_create(
        self,
        concept: str,
    ) -> int:
        existing = (
            self.concept_id_by_name.get(
                concept
            )
        )

        if existing is not None:
            return existing

        identifier = len(
            self.concept_id_by_name
        )

        self.concept_id_by_name[
            concept
        ] = identifier

        self.concept_name_by_id[
            identifier
        ] = concept

        return identifier

    def add_edge(
        self,
        word: str,
        relation: str,
        concept: str,
    ) -> None:
        concept = concept.strip().lower()

        if not concept:
            return

        identifier = self.get_or_create(
            concept
        )

        targets = self.edges[
            word
        ][
            relation
        ]

        if identifier not in targets:
            targets.append(identifier)
            self.edge_count += 1

    def concepts(
        self,
        word: str,
        relation: str,
    ) -> list[str]:
        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier
            in self.edges.get(
                word,
                {},
            ).get(
                relation,
                [],
            )
        ]

    def save(
        self,
        path: Path,
    ) -> None:
        serial_edges = {
            word: {
                relation: identifiers
                for relation, identifiers
                in relation_map.items()
            }
            for word, relation_map
            in self.edges.items()
        }

        payload = {
            "concept_id_by_name": self.concept_id_by_name,
            "edges": serial_edges,
            "edge_count": self.edge_count,
            "relations": RELATIONS,
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(
    word: str,
    seed: SemanticSeed,
    lexical_index: dict[str, list[str]],
) -> list[str]:
    """
    Candidate retrieval is deliberately independent of the target relation.

    Relation typing comes from the LLM.
    Candidate identity comes from the semantic seed / lexical context.
    """
    scores = Counter()

    # Direct cue evidence.
    for feature, weight in (
        seed.cue_features.get(
            word,
            Counter(),
        ).items()
    ):
        scores[
            feature
        ] += (
            4.0
            + math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

    # Lexical-neighbor evidence.
    for neighbor in lexical_index.get(
        word,
        [],
    ):
        for feature, weight in (
            seed.cue_features.get(
                neighbor,
                Counter(),
            ).most_common(10)
        ):
            scores[
                feature
            ] += (
                1.0
                + 0.25
                * math.log1p(
                    max(
                        0.0,
                        weight,
                    )
                )
            )

    # Global fallback.
    for feature, weight in seed.feature_weight.most_common(
        GLOBAL_FALLBACK
    ):
        scores[
            feature
        ] += (
            0.5
            * math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

    ranked = [
        feature
        for feature, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]

    return ranked


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has no PAD/EOS token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    print(
        "device:",
        device,
        flush=True,
    )

    if device.type == "cuda":
        print(
            "gpu:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    return (
        tokenizer,
        model,
        device,
    )


def apply_chat(
    tokenizer,
    system: str,
    user: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            system
            + "\n\n"
            + user
            + "\n\nAssistant:"
        )


# ---------------------------------------------------------------------------
# Typed selection prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a semantic relation selector. "
    "Choose ONLY candidate numbers. "
    "Return numbers, not words. "
    "No explanation."
)


def relation_prompt(
    tokenizer,
    word: str,
    relation: str,
    candidates: list[str],
) -> str:
    numbered = "\n".join(
        f"{index + 1}. {concept}"
        for index, concept
        in enumerate(candidates)
    )

    relation_questions = {
        "CATEGORY": (
            f"What category does {word} belong to?"
        ),
        "PROPERTY": (
            f"What properties describe {word}?"
        ),
        "ACTION": (
            f"What action or behavior is associated with {word}?"
        ),
        "USE": (
            f"What is {word} used for?"
        ),
        "PART": (
            f"What part or component is associated with {word}?"
        ),
        "RELATION": (
            f"What concept is {word} related to?"
        ),
    }

    return apply_chat(
        tokenizer,
        SYSTEM_PROMPT,
        f"""
WORD:
{word}

RELATION:
{relation}

QUESTION:
{relation_questions[relation]}

CANDIDATES:
{numbered}

Select up to {MAX_SELECTED} candidates.

Return ONLY space-separated numbers.

Example:
2 7 13

Rules:
- Every number must refer to a candidate.
- Never output words.
- Never output explanations.
- If none fit, output 0.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_indices(
    text: str,
    candidate_count: int,
) -> list[int]:
    numbers = re.findall(
        r"\b\d+\b",
        text,
    )

    result = []

    for raw in numbers:
        index = int(raw)

        if index == 0:
            continue

        if not (
            1 <= index <= candidate_count
        ):
            continue

        zero_based = index - 1

        if zero_based not in result:
            result.append(
                zero_based
            )

        if len(result) >= MAX_SELECTED:
            break

    return result


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def f1(
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
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def overall_word_f1(
    predictions: dict[
        tuple[str, str],
        list[str],
    ],
    seed: SemanticSeed,
    words: list[str],
) -> float:
    values = []

    for word in words:
        predicted = set()

        for relation in RELATIONS:
            predicted.update(
                predictions.get(
                    (
                        word,
                        relation,
                    ),
                    [],
                )
            )

        gold = set(
            feature
            for feature, _weight
            in seed.cue_features.get(
                word,
                Counter(),
            ).most_common(8)
        )

        if not gold:
            continue

        values.append(
            f1(
                predicted,
                gold,
            )
        )

    return (
        sum(values)
        / max(
            1,
            len(values),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V122 TYPED SEMANTIC GRAPH LEARNING ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    lexical_index = build_lexical_index(
        words
    )

    seed = SemanticSeed()
    seed.load(
        SEMANTICS_PATH
    )

    tokenizer, model, device = (
        load_model()
    )

    graph = TypedGraph()

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "relations:",
        ", ".join(RELATIONS),
    )

    print(
        "candidate_limit:",
        CANDIDATE_LIMIT,
    )

    print(
        "total_queries:",
        len(words) * len(RELATIONS),
    )

    # ---------------------------------------------------------------
    # Cache candidate sets so the six relation queries for a word use
    # exactly the same semantic candidate universe.
    # ---------------------------------------------------------------

    candidate_cache = {
        word: retrieve_candidates(
            word,
            seed,
            lexical_index,
        )
        for word in words
    }

    # ---------------------------------------------------------------
    # Batched typed queries.
    # ---------------------------------------------------------------

    jobs = [
        (
            word,
            relation,
            candidate_cache[word],
        )
        for word in words
        for relation in RELATIONS
    ]

    query_count = len(jobs)

    accepted = Counter()
    invalid = Counter()
    selected_count = Counter()

    per_relation_predictions: dict[
        str,
        dict[str, list[str]],
    ] = {
        relation: {}
        for relation in RELATIONS
    }

    traces = []

    for start in range(
        0,
        query_count,
        BATCH_SIZE,
    ):
        batch = jobs[
            start:start + BATCH_SIZE
        ]

        prompts = [
            relation_prompt(
                tokenizer,
                word,
                relation,
                candidates,
            )
            for word, relation, candidates
            in batch
        ]

        raw_outputs = generate_batch(
            tokenizer,
            model,
            device,
            prompts,
        )

        for (
            (word, relation, candidates),
            raw,
        ) in zip(
            batch,
            raw_outputs,
        ):
            indices = parse_indices(
                raw,
                len(candidates),
            )

            concepts = [
                candidates[index]
                for index in indices
            ]

            per_relation_predictions[
                relation
            ][
                word
            ] = concepts

            if indices:
                accepted[
                    relation
                ] += 1

                selected_count[
                    relation
                ] += len(indices)
            else:
                invalid[
                    relation
                ] += 1

            # Store typed graph edges.
            for concept in concepts:
                graph.add_edge(
                    word,
                    relation,
                    concept,
                )

            if word in TRACE_WORDS:
                traces.append(
                    {
                        "word": word,
                        "relation": relation,
                        "candidates": candidates,
                        "raw": raw,
                        "selected": concepts,
                    }
                )

        processed = min(
            start + BATCH_SIZE,
            query_count,
        )

        if (
            processed <= BATCH_SIZE
            or processed % PRINT_EVERY == 0
            or processed == query_count
        ):
            print(
                f"QUERY "
                f"{processed:5d}/{query_count:5d} "
                f"edges={graph.edge_count} "
                f"concepts={len(graph.concept_id_by_name)}",
                flush=True,
            )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    relation_f1 = {}

    for relation in RELATIONS:
        values = []

        for word in words:
            predicted = set(
                per_relation_predictions[
                    relation
                ].get(
                    word,
                    [],
                )
            )

            gold = set(
                feature
                for feature, _weight
                in seed.cue_features.get(
                    word,
                    Counter(),
                ).most_common(8)
            )

            if not gold:
                continue

            values.append(
                f1(
                    predicted,
                    gold,
                )
            )

        relation_f1[
            relation
        ] = (
            sum(values)
            / max(
                1,
                len(values),
            )
        )

    flat_predictions = {}

    for relation in RELATIONS:
        for word, concepts in (
            per_relation_predictions[
                relation
            ].items()
        ):
            key = (
                word,
                relation,
            )

            flat_predictions[
                key
            ] = concepts

    flat_f1 = overall_word_f1(
        flat_predictions,
        seed,
        words,
    )

    # ---------------------------------------------------------------
    # Statistics on relation overlap / graph shape.
    # ---------------------------------------------------------------

    relation_edge_counts = {
        relation: sum(
            len(
                per_relation_predictions[
                    relation
                ].get(
                    word,
                    [],
                )
            )
            for word in words
        )
        for relation in RELATIONS
    }

    words_with_relation = {
        relation: sum(
            bool(
                per_relation_predictions[
                    relation
                ].get(
                    word,
                    [],
                )
            )
            for word in words
        )
        for relation in RELATIONS
    }

    # ---------------------------------------------------------------
    # Save graph.
    # ---------------------------------------------------------------

    graph_payload = {
        "experiment": (
            "V122 typed semantic graph learning"
        ),
        "relations": RELATIONS,
        "concept_id_by_name": (
            graph.concept_id_by_name
        ),
        "edges": {
            word: {
                relation: identifiers
                for relation, identifiers
                in relation_map.items()
            }
            for word, relation_map
            in graph.edges.items()
        },
        "edge_count": graph.edge_count,
    }

    OUTPUT_GRAPH.write_text(
        json.dumps(
            graph_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "experiment": (
            "V122 typed semantic graph learning"
        ),
        "dictionary_words": len(words),
        "relations": RELATIONS,
        "candidate_limit": CANDIDATE_LIMIT,
        "query_count": query_count,
        "final_concepts": len(
            graph.concept_id_by_name
        ),
        "edge_count": graph.edge_count,
        "accepted": dict(accepted),
        "invalid": dict(invalid),
        "selected_count": dict(selected_count),
        "relation_edge_counts": relation_edge_counts,
        "words_with_relation": words_with_relation,
        "relation_f1": relation_f1,
        "overall_flat_f1": flat_f1,
        "traces": traces,
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== V122 SUMMARY ==="
    )

    print(
        "final_concepts:",
        len(
            graph.concept_id_by_name
        ),
    )

    print(
        "edge_count:",
        graph.edge_count,
    )

    print(
        "relation_edge_counts:",
        relation_edge_counts,
    )

    print(
        "words_with_relation:",
        words_with_relation,
    )

    print(
        "relation_f1:",
        relation_f1,
    )

    print(
        "overall_flat_f1:",
        flat_f1,
    )

    print()
    print(
        "=== TRACE ==="
    )

    for trace in traces[:120]:
        print(
            json.dumps(
                trace,
                ensure_ascii=False,
            )
        )

    print()
    print(
        "saved_graph:",
        OUTPUT_GRAPH,
    )

    print(
        "saved_report:",
        OUTPUT_REPORT,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V122 COMPLETE ==="
    )


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_batch(
    tokenizer,
    model,
    device,
    prompts: list[str],
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []

    for row in range(
        output.shape[0]
    ):
        prompt_len = int(
            attention_mask[row].sum().item()
        )

        results.append(
            tokenizer.decode(
                output[
                    row,
                    prompt_len:,
                ],
                skip_special_tokens=True,
            ).strip()
        )

    return results


if __name__ == "__main__":
    main()
