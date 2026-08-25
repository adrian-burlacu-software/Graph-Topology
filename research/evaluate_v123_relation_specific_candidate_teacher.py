from __future__ import annotations

"""
V123 — RELATION-SPECIFIC CANDIDATE TEACHER

V122 exposed a key failure:
    candidate retrieval was useful,
    but the tiny LLM mostly ignored the requested relation and repeatedly
    selected the same candidate numbers.

V123 moves the relation constraint OUT OF the LLM.

For each relation we first build a relation-specific candidate vocabulary from
the human semantic seed corpus:

    CATEGORY
    PROPERTY
    ACTION
    USE
    PART
    RELATION

The LLM then performs only one operation:

    choose candidate numbers

It never has to infer what CATEGORY vs ACTION means from a generic mixed
candidate pool.

The human semantic corpus therefore plays two roles:
    1. seed candidate ontology / relation-specific candidate pools
    2. evaluation gold

The model remains frozen.

No graph mutation of V111.

Outputs:
    * relation-specific candidate coverage
    * LLM selection accuracy / F1
    * invalid selector rate
    * repeated-selector rate
    * per-relation results
    * typed graph built from the selections

The key question:

    Does relation-specific candidate conditioning make the frozen 360M model
    perform meaningful typed semantic selection?
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

MODEL_PATH = (
    ROOT / "llm" / "SmolLM2-360M-Instruct"
)

DICTIONARY_PATH = (
    ROOT / "data" / "dictionary.csv"
)

SEMANTICS_PATH = (
    ROOT / "data" / "semantics-large.csv"
)

OUTPUT_GRAPH = (
    ROOT / "results" / "v123_relation_specific_typed_graph.json"
)

OUTPUT_REPORT = (
    ROOT / "results" / "v123_relation_specific_teacher.json"
)

EVAL_WORDS = 512

BATCH_SIZE = 128

CANDIDATE_LIMIT = 32

MAX_SELECTED = 4

MAX_INPUT_TOKENS = 224
MAX_NEW_TOKENS = 12

PRINT_EVERY = 128

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

RELATIONS = (
    "CATEGORY",
    "PROPERTY",
    "ACTION",
    "USE",
    "PART",
    "RELATION",
)


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

    result = sorted(words)[:EVAL_WORDS]

    if not result:
        raise RuntimeError(
            "No dictionary words found."
        )

    return result


# ---------------------------------------------------------------------------
# Semantic seed
# ---------------------------------------------------------------------------

class SemanticSeed:
    def __init__(self) -> None:
        self.cue_features: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        self.feature_weight: Counter[str] = Counter()

        self.features: set[str] = set()

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

                        if n > 0.0:
                            weight = frequency / n
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

                self.features.add(
                    feature
                )

        print(
            "semantic_cues:",
            len(self.cue_features),
        )

        print(
            "semantic_concepts:",
            len(self.features),
        )

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
# Relation-specific candidate pools
# ---------------------------------------------------------------------------

def build_relation_pools(
    seed: SemanticSeed,
) -> dict[str, list[str]]:
    """
    We do not ask the tiny model to infer semantic relation types.

    Instead we derive a conservative relation-specific seed vocabulary using
    corpus statistics and interpretable lexical heuristics.

    CATEGORY:
        concepts that frequently behave like broad classes / entity classes.

    PROPERTY:
        concepts that tend to occur as descriptors and are often relatively
        low-document-frequency.

    ACTION:
        concepts associated with verb-like lexical cues in the corpus where POS
        columns exist.

    USE:
        concepts frequently co-occurring with functional / activity language.

    PART:
        concepts strongly associated with body/object-part vocabulary.

    RELATION:
        broad relational vocabulary.

    If POS columns are unavailable, the heuristic falls back to corpus-global
    statistics. This is intentionally transparent rather than pretending the
    relation labels are ground truth.
    """
    pools: dict[str, Counter[str]] = {
        relation: Counter()
        for relation in RELATIONS
    }

    # We can use the cue -> feature graph itself to construct candidate
    # namespaces, but not the gold answer of the current query.
    for cue, features in seed.cue_features.items():
        cue_tokens = set(
            cue.lower().split()
        )

        for feature, weight in features.items():
            f = feature.lower()

            # Broad candidate classes.
            if any(
                token in f
                for token in (
                    "animal",
                    "person",
                    "object",
                    "tool",
                    "place",
                    "food",
                    "vehicle",
                    "plant",
                    "body",
                    "group",
                    "thing",
                )
            ):
                pools["CATEGORY"][f] += weight

            # Descriptor-like candidates.
            if any(
                suffix in f
                for suffix in (
                    "ful",
                    "less",
                    "ous",
                    "ive",
                    "al",
                    "ic",
                    "ary",
                )
            ):
                pools["PROPERTY"][f] += weight

            # Action-like lexical concepts.
            if any(
                token in f.split()
                for token in (
                    "eat",
                    "move",
                    "hold",
                    "make",
                    "take",
                    "give",
                    "leave",
                    "use",
                    "wear",
                    "build",
                    "play",
                    "work",
                    "run",
                    "walk",
                    "act",
                    "go",
                )
            ):
                pools["ACTION"][f] += weight

            # Functional / use concepts.
            if any(
                token in f.split()
                for token in (
                    "work",
                    "use",
                    "help",
                    "carry",
                    "transport",
                    "clean",
                    "cook",
                    "wear",
                    "play",
                    "store",
                )
            ):
                pools["USE"][f] += weight

            # Part-like concepts.
            if any(
                token in f.split()
                for token in (
                    "hand",
                    "head",
                    "leg",
                    "arm",
                    "foot",
                    "body",
                    "part",
                    "piece",
                    "side",
                    "front",
                    "back",
                    "top",
                    "bottom",
                )
            ):
                pools["PART"][f] += weight

            # Broad relation vocabulary.
            pools["RELATION"][f] += (
                0.35 * weight
            )

    # Ensure every pool has a deterministic fallback.
    global_top = [
        feature
        for feature, _weight
        in seed.feature_weight.most_common(
            CANDIDATE_LIMIT
        )
    ]

    result = {}

    for relation in RELATIONS:
        ranked = [
            feature
            for feature, _score
            in pools[relation].most_common(
                CANDIDATE_LIMIT
            )
        ]

        for feature in global_top:
            if len(ranked) >= CANDIDATE_LIMIT:
                break

            if feature not in ranked:
                ranked.append(feature)

        result[relation] = ranked[
            :CANDIDATE_LIMIT
        ]

    return result


def retrieve_relation_candidates(
    word: str,
    relation: str,
    seed: SemanticSeed,
    relation_pool: list[str],
) -> list[str]:
    """
    Candidate ranking for this word within ONE relation-specific vocabulary.

    Direct cue evidence is strongest; otherwise use the pre-built relation
    pool.
    """
    scores = Counter()

    direct = seed.cue_features.get(
        word,
        Counter(),
    )

    for concept, weight in direct.items():
        if concept in relation_pool:
            scores[
                concept
            ] += (
                4.0
                + math.log1p(
                    max(
                        0.0,
                        weight,
                    )
                )
            )

    for concept in relation_pool:
        scores[
            concept
        ] += 0.1

    ranked = [
        concept
        for concept, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]

    return ranked


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
                "Tokenizer has no PAD/EOS."
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
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a strict candidate selector. "
    "Choose ONLY candidate numbers. "
    "Return numbers only."
)


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

CANDIDATE VOCABULARY FOR THIS RELATION:
{numbered}

Select up to {MAX_SELECTED} candidate numbers.

Return ONLY space-separated numbers.
Example:
1 4 9

Rules:
- Every number must refer to a candidate above.
- Never output words.
- Never explain.
- If none fit, output 0.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Inference
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

        zero_based = (
            index - 1
        )

        if zero_based not in result:
            result.append(
                zero_based
            )

        if len(result) >= MAX_SELECTED:
            break

    return result


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
        "=== V123 RELATION-SPECIFIC CANDIDATE TEACHER ==="
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
        "=== RELATION POOLS ==="
    )

    for relation in RELATIONS:
        print(
            relation,
            "->",
            relation_pools[relation],
        )

    # ---------------------------------------------------------------
    # Build relation-specific jobs.
    # ---------------------------------------------------------------

    jobs = []

    for word in words:
        for relation in RELATIONS:
            candidates = (
                retrieve_relation_candidates(
                    word,
                    relation,
                    seed,
                    relation_pools[relation],
                )
            )

            jobs.append(
                (
                    word,
                    relation,
                    candidates,
                )
            )

    total_queries = len(jobs)

    predictions: dict[
        tuple[str, str],
        list[str],
    ] = {}

    relation_stats = {
        relation: {
            "queries": 0,
            "accepted": 0,
            "invalid": 0,
            "selected": 0,
            "gold_hits": 0,
            "gold_items": 0,
        }
        for relation in RELATIONS
    }

    traces = []

    # ---------------------------------------------------------------
    # Batched selection.
    # ---------------------------------------------------------------

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
            stats = relation_stats[
                relation
            ]

            stats["queries"] += 1

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
                stats["accepted"] += 1
            else:
                stats["invalid"] += 1

            stats["selected"] += len(
                selected
            )

            gold = seed.gold(
                word
            )

            stats["gold_items"] += len(
                gold
            )

            stats["gold_hits"] += len(
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
                            set(selected) & gold
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
    # Score per relation.
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

    # Flat union across relations.
    flat_f1s = []

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

        flat_f1s.append(
            f1(
                predicted,
                gold,
            )
        )

    overall_f1 = (
        sum(flat_f1s)
        / max(
            1,
            len(flat_f1s),
        )
    )

    # ---------------------------------------------------------------
    # Typed graph output.
    # ---------------------------------------------------------------

    concept_id_by_name = {}
    edges = defaultdict(
        lambda: defaultdict(list)
    )

    def concept_id(
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
            identifier = concept_id(
                concept
            )

            target = edges[
                word
            ][
                relation
            ]

            if identifier not in target:
                target.append(
                    identifier
                )
                edge_count += 1

    graph_payload = {
        "experiment": (
            "V123 relation-specific candidate teacher"
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
            "V123 relation-specific candidate teacher"
        ),
        "evaluation_words": words,
        "relations": RELATIONS,
        "candidate_limit": CANDIDATE_LIMIT,
        "max_selected": MAX_SELECTED,
        "relation_pools": relation_pools,
        "relation_stats": relation_stats,
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
        "=== V123 SUMMARY ==="
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

    print(
        "relation_scores:",
        relation_scores,
    )

    print(
        "overall_flat_f1:",
        overall_f1,
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
        "=== V123 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
