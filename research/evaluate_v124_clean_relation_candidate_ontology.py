from __future__ import annotations

"""
V124 — CLEAN RELATION-SPECIFIC CANDIDATE ONTOLOGY

V123 showed a large improvement for ACTION / USE / PART / RELATION when the
candidate vocabulary was relation-specific, but CATEGORY and PROPERTY were
still contaminated by generic concepts.

V124 keeps the exact same selection benchmark while making the relation
candidate pools substantially cleaner and more explicit.

Important:
    * Frozen SmolLM2-360M-Instruct
    * No graph mutation
    * No free-form semantic generation
    * LLM returns candidate NUMBERS only
    * semantics-large.csv supplies seed vocabulary + evaluation gold
    * Same dictionary and same 6 typed relations

Relation pools are built from the semantic corpus using POS information when
available, plus transparent lexical heuristics:

    CATEGORY
        noun-like class/entity concepts and broad class vocabulary

    PROPERTY
        adjective-like / descriptive concepts

    ACTION
        verb-like concepts

    USE
        verb/action concepts associated with function, plus functional nouns

    PART
        body/object-part vocabulary and noun-like components

    RELATION
        broad relation/connective concepts

The key goal is NOT perfect ontology induction. It is to remove the worst
cross-relation contamination so the frozen selector can actually use the
typed query.

Outputs:
    v124_clean_relation_candidate_ontology.json

No V111/V122/V123 graph is modified.
"""

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "llm" / "SmolLM2-360M-Instruct"
DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

OUTPUT_GRAPH = (
    ROOT / "results" / "v124_clean_typed_semantic_graph.json"
)
OUTPUT_REPORT = (
    ROOT / "results" / "v124_clean_relation_candidate_ontology.json"
)

EVAL_WORDS = 512
BATCH_SIZE = 128

CANDIDATE_LIMIT = 32
MAX_SELECTED = 4

MAX_INPUT_TOKENS = 224
MAX_NEW_TOKENS = 12

PRINT_EVERY = 128

RELATIONS = (
    "CATEGORY",
    "PROPERTY",
    "ACTION",
    "USE",
    "PART",
    "RELATION",
)

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

# Transparent broad semantic seeds used only to improve candidate-pool purity.
CATEGORY_SEEDS = {
    "animal",
    "person",
    "people",
    "human",
    "object",
    "thing",
    "tool",
    "device",
    "vehicle",
    "place",
    "food",
    "plant",
    "body",
    "material",
    "substance",
    "group",
    "organization",
    "building",
    "part",
}

USE_SEEDS = {
    "use",
    "work",
    "transport",
    "carry",
    "clean",
    "cook",
    "wear",
    "play",
    "store",
    "hold",
    "help",
    "protect",
    "move",
}

PART_SEEDS = {
    "part",
    "piece",
    "body",
    "head",
    "hand",
    "arm",
    "leg",
    "foot",
    "face",
    "eye",
    "ear",
    "mouth",
    "finger",
    "side",
    "top",
    "bottom",
    "front",
    "back",
    "inside",
    "outside",
}

RELATION_SEEDS = {
    "cause",
    "effect",
    "relation",
    "relationship",
    "similar",
    "opposite",
    "same",
    "different",
    "type",
    "kind",
    "part",
    "whole",
    "person",
    "place",
    "thing",
}


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def load_dictionary(path: Path) -> list[str]:
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

    result = sorted(words)[:EVAL_WORDS]

    if not result:
        raise RuntimeError("No dictionary words found.")

    return result


# ---------------------------------------------------------------------------
# Semantic corpus
# ---------------------------------------------------------------------------

class SemanticSeed:
    def __init__(self) -> None:
        self.cue_features: dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_weight: Counter[str] = Counter()

        # feature -> observed POS distribution from the corpus
        self.feature_pos: dict[str, Counter[str]] = defaultdict(Counter)

        self.pos_feature_weight: dict[str, Counter[str]] = defaultdict(Counter)

    def load(self, path: Path) -> None:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle)

            fields = set(reader.fieldnames or [])

            for row in reader:
                cue = row.get("cue", "").strip().lower()
                feature = row.get("translated", "").strip().lower()

                if not cue or not feature:
                    continue

                try:
                    weight = float(
                        row.get(
                            "normalized_translated",
                            0.0,
                        )
                    )
                except (TypeError, ValueError):
                    weight = 0.0

                if weight <= 0.0:
                    try:
                        frequency = float(
                            row.get(
                                "frequency_translated",
                                0.0,
                            )
                        )
                        n = float(row.get("n", 0.0))
                        if n > 0:
                            weight = frequency / n
                    except (TypeError, ValueError):
                        weight = 0.0

                if weight <= 0.0:
                    continue

                self.cue_features[cue][feature] += weight
                self.feature_weight[feature] += weight

                # Semantics-large contains POS columns in the corpus used for
                # the previous experiments. Use whichever exists.
                pos = (
                    row.get("pos_feature")
                    or row.get("pos_translated")
                    or ""
                ).strip().lower()

                if pos:
                    self.feature_pos[feature][pos] += weight
                    self.pos_feature_weight[pos][feature] += weight

    def gold(self, word: str, limit: int = 8) -> set[str]:
        return {
            feature
            for feature, _weight
            in self.cue_features.get(word, Counter()).most_common(limit)
        }


# ---------------------------------------------------------------------------
# Relation-specific candidate pools
# ---------------------------------------------------------------------------

def pos_score(
    seed: SemanticSeed,
    concept: str,
    allowed: set[str],
) -> float:
    if concept in allowed:
        return 8.0

    score = 0.0
    observed = seed.feature_pos.get(
        concept,
        Counter(),
    )

    for pos, weight in observed.items():
        if pos in allowed:
            score += 2.0 * math.log1p(max(0.0, weight))

    return score


def lexical_property_score(
    concept: str,
) -> float:
    score = 0.0

    for suffix in (
        "y",
        "ful",
        "less",
        "ous",
        "ive",
        "al",
        "ic",
        "able",
        "ible",
        "ary",
        "ish",
        "ed",
        "ing",
    ):
        if concept.endswith(suffix):
            score += 1.5

    return score


def lexical_action_score(
    concept: str,
) -> float:
    score = 0.0

    for suffix in (
        "ate",
        "ify",
        "ise",
        "ize",
        "en",
        "ing",
    ):
        if concept.endswith(suffix):
            score += 1.2

    # Common atomic actions.
    if concept in {
        "act",
        "move",
        "eat",
        "give",
        "take",
        "hold",
        "leave",
        "go",
        "run",
        "walk",
        "make",
        "build",
        "play",
        "wear",
        "clean",
        "carry",
        "use",
        "work",
        "help",
        "protect",
        "open",
        "close",
        "cut",
        "push",
        "pull",
    }:
        score += 4.0

    return score


def build_relation_pools(
    seed: SemanticSeed,
) -> dict[str, list[str]]:
    pools: dict[str, Counter[str]] = {
        relation: Counter()
        for relation in RELATIONS
    }

    all_features = list(seed.feature_weight)

    # POS aliases used by the corpus.
    noun_pos = {
        "noun",
        "proper_noun",
    }

    adjective_pos = {
        "adjective",
    }

    verb_pos = {
        "verb",
    }

    for concept in all_features:
        base = math.log1p(
            max(
                0.0,
                seed.feature_weight.get(
                    concept,
                    0.0,
                ),
            )
        )

        pos = seed.feature_pos.get(
            concept,
            Counter(),
        )

        noun_signal = sum(
            weight
            for label, weight
            in pos.items()
            if label in noun_pos
        )

        adjective_signal = sum(
            weight
            for label, weight
            in pos.items()
            if label in adjective_pos
        )

        verb_signal = sum(
            weight
            for label, weight
            in pos.items()
            if label in verb_pos
        )

        # CATEGORY:
        # noun-heavy + explicit class seeds, but explicitly suppress obvious
        # action-only terms.
        category_score = (
            base
            + 1.5 * math.log1p(noun_signal)
            + (7.0 if concept in CATEGORY_SEEDS else 0.0)
        )

        if concept in {
            "eat",
            "move",
            "give",
            "take",
            "act",
            "hold",
            "leave",
            "wear",
        }:
            category_score -= 10.0

        pools["CATEGORY"][concept] = category_score

        # PROPERTY:
        property_score = (
            base
            + 2.0 * math.log1p(adjective_signal)
            + lexical_property_score(concept)
        )

        # Suppress obvious physical objects / actions from PROPERTY.
        if concept in CATEGORY_SEEDS:
            property_score -= 6.0

        pools["PROPERTY"][concept] = property_score

        # ACTION:
        action_score = (
            base
            + 2.2 * math.log1p(verb_signal)
            + lexical_action_score(concept)
        )

        pools["ACTION"][concept] = action_score

        # USE:
        use_score = (
            0.7 * base
            + 1.0 * math.log1p(verb_signal)
            + (
                6.0
                if concept in USE_SEEDS
                else 0.0
            )
        )

        # Things can also be uses / functions.
        if concept in {
            "food",
            "transport",
            "work",
            "cleaning",
            "cooking",
            "play",
            "storage",
        }:
            use_score += 3.0

        pools["USE"][concept] = use_score

        # PART:
        part_score = (
            base
            + 1.8 * math.log1p(noun_signal)
            + (
                7.0
                if concept in PART_SEEDS
                else 0.0
            )
        )

        pools["PART"][concept] = part_score

        # RELATION:
        relation_score = (
            0.5 * base
            + (
                6.0
                if concept in RELATION_SEEDS
                else 0.0
            )
        )

        pools["RELATION"][concept] = relation_score

    result = {}

    for relation in RELATIONS:
        ranked = [
            concept
            for concept, _score
            in pools[relation].most_common(
                CANDIDATE_LIMIT
            )
        ]

        # Ensure deterministic minimum vocabulary size.
        if len(ranked) < CANDIDATE_LIMIT:
            for concept in seed.feature_weight.most_common():
                feature = concept[0]
                if feature not in ranked:
                    ranked.append(feature)
                if len(ranked) >= CANDIDATE_LIMIT:
                    break

        result[relation] = ranked[:CANDIDATE_LIMIT]

    return result


def relation_candidates_for_word(
    word: str,
    relation: str,
    seed: SemanticSeed,
    pool: list[str],
) -> list[str]:
    """
    Rank ONLY inside the relation-specific pool.

    This is the same basic protocol as V123, so the comparison is clean.
    """
    scores = Counter()

    direct = seed.cue_features.get(
        word,
        Counter(),
    )

    for concept, weight in direct.items():
        if concept not in pool:
            continue

        scores[concept] += (
            4.0
            + math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

    # Keep the relation-specific vocabulary even when no direct match exists.
    for concept in pool:
        scores[concept] += 0.1

    return [
        concept
        for concept, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
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

        tokenizer.pad_token = tokenizer.eos_token

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

    return tokenizer, model, device


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
# Selector
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a strict semantic candidate selector. "
    "Select only candidate numbers. "
    "Return numbers only."
)


def relation_question(
    word: str,
    relation: str,
) -> str:
    return {
        "CATEGORY": (
            f"What category does {word} belong to?"
        ),
        "PROPERTY": (
            f"What property or quality describes {word}?"
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
    }[relation]


def make_prompt(
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

    return apply_chat(
        tokenizer,
        SYSTEM_PROMPT,
        f"""
WORD:
{word}

RELATION:
{relation}

QUESTION:
{relation_question(word, relation)}

CANDIDATE VOCABULARY:
{numbered}

Select up to {MAX_SELECTED} candidates.

Return ONLY space-separated candidate numbers.
Example:
2 7 13

If none fit, return 0.

Never output candidate words.
Never explain.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Inference
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
# Scoring
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

    precision = hits / len(
        predicted
    )

    recall = hits / len(
        gold
    )

    if precision + recall == 0.0:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V124 CLEAN RELATION-SPECIFIC CANDIDATE ONTOLOGY ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    seed = SemanticSeed()
    seed.load(
        SEMANTICS_PATH
    )

    relation_pools = build_relation_pools(
        seed
    )

    tokenizer, model, device = load_model()

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "total_queries:",
        len(words) * len(RELATIONS),
    )

    print()
    print(
        "=== CLEAN RELATION POOLS ==="
    )

    pool_stats = {}

    for relation in RELATIONS:
        pool = relation_pools[relation]

        pool_stats[relation] = {
            "size": len(pool),
            "items": pool,
        }

        print(
            relation,
            "=>",
            pool,
        )

    # ---------------------------------------------------------------
    # Prepare jobs.
    # ---------------------------------------------------------------

    jobs = []

    for word in words:
        for relation in RELATIONS:
            pool = relation_pools[
                relation
            ]

            candidates = relation_candidates_for_word(
                word,
                relation,
                seed,
                pool,
            )

            jobs.append(
                (
                    word,
                    relation,
                    candidates,
                )
            )

    predictions = {}

    stats = {
        relation: {
            "queries": 0,
            "accepted": 0,
            "invalid": 0,
            "selected": 0,
            "gold_hits": 0,
        }
        for relation in RELATIONS
    }

    traces = []

    # ---------------------------------------------------------------
    # Batched selection.
    # ---------------------------------------------------------------

    total_queries = len(jobs)

    for start in range(
        0,
        total_queries,
        BATCH_SIZE,
    ):
        batch = jobs[
            start:start + BATCH_SIZE
        ]

        prompts = [
            make_prompt(
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
            relation_stat = stats[
                relation
            ]

            relation_stat["queries"] += 1

            indices = parse_indices(
                raw,
                len(candidates),
            )

            selected = [
                candidates[index]
                for index in indices
            ]

            predictions[
                (
                    word,
                    relation,
                )
            ] = selected

            if selected:
                relation_stat["accepted"] += 1
                relation_stat["selected"] += len(
                    selected
                )
            else:
                relation_stat["invalid"] += 1

            gold = seed.gold(
                word
            )

            relation_stat["gold_hits"] += len(
                set(selected)
                & gold
            )

            if word in TRACE_WORDS:
                traces.append(
                    {
                        "word": word,
                        "relation": relation,
                        "candidates": candidates,
                        "raw": raw,
                        "selected": selected,
                        "gold": sorted(gold),
                        "hits": sorted(
                            set(selected)
                            & gold
                        ),
                    }
                )

        processed = min(
            start + BATCH_SIZE,
            total_queries,
        )

        if (
            processed <= BATCH_SIZE
            or processed % PRINT_EVERY == 0
            or processed == total_queries
        ):
            print(
                f"QUERY "
                f"{processed:5d}/{total_queries:5d}",
                flush=True,
            )

    # ---------------------------------------------------------------
    # Relation scores.
    # ---------------------------------------------------------------

    relation_scores = {}

    for relation in RELATIONS:
        precisions = []
        recalls = []
        f1s = []

        for word in words:
            predicted = set(
                predictions.get(
                    (
                        word,
                        relation,
                    ),
                    [],
                )
            )

            gold = seed.gold(
                word
            )

            if not gold:
                continue

            hits = len(
                predicted
                & gold
            )

            precision = (
                hits
                / max(
                    1,
                    len(predicted),
                )
            )

            recall = (
                hits
                / max(
                    1,
                    len(gold),
                )
            )

            precisions.append(
                precision
            )

            recalls.append(
                recall
            )

            f1s.append(
                f1(
                    predicted,
                    gold,
                )
            )

        relation_scores[
            relation
        ] = {
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

    # Flat union.
    flat_scores = []

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

        gold = seed.gold(
            word
        )

        if not gold:
            continue

        flat_scores.append(
            f1(
                predicted,
                gold,
            )
        )

    overall_f1 = (
        sum(flat_scores)
        / max(
            1,
            len(flat_scores),
        )
    )

    # ---------------------------------------------------------------
    # Typed graph output.
    # ---------------------------------------------------------------

    concept_id_by_name = {}
    edges = defaultdict(
        lambda: defaultdict(list)
    )

    def get_id(
        concept: str,
    ) -> int:
        if concept not in concept_id_by_name:
            concept_id_by_name[
                concept
            ] = len(
                concept_id_by_name
            )

        return concept_id_by_name[
            concept
        ]

    edge_count = 0

    for (
        word,
        relation,
    ), selected in predictions.items():
        for concept in selected:
            identifier = get_id(
                concept
            )

            if identifier not in edges[
                word
            ][
                relation
            ]:
                edges[
                    word
                ][
                    relation
                ].append(
                    identifier
                )
                edge_count += 1

    graph_payload = {
        "experiment": (
            "V124 clean relation-specific candidate ontology"
        ),
        "relations": RELATIONS,
        "concept_id_by_name": concept_id_by_name,
        "edges": {
            word: {
                relation: identifiers
                for relation, identifiers
                in relation_map.items()
            }
            for word, relation_map
            in edges.items()
        },
        "edge_count": edge_count,
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
            "V124 clean relation-specific candidate ontology"
        ),
        "evaluation_words": words,
        "relations": RELATIONS,
        "candidate_limit": CANDIDATE_LIMIT,
        "max_selected": MAX_SELECTED,
        "pool_stats": pool_stats,
        "relation_stats": stats,
        "relation_scores": relation_scores,
        "overall_flat_f1": overall_f1,
        "graph_concepts": len(
            concept_id_by_name
        ),
        "graph_edges": edge_count,
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
        "=== V124 SUMMARY ==="
    )

    print(
        "relation_scores:",
        relation_scores,
    )

    print(
        "overall_flat_f1:",
        overall_f1,
    )

    print(
        "graph_concepts:",
        len(
            concept_id_by_name
        ),
    )

    print(
        "graph_edges:",
        edge_count,
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
        "=== V124 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
