from __future__ import annotations

"""
V120 — CONSTRAINED GRAPH -> LLM SEMANTIC SELECTION

This is the next clean experiment.

Do NOT ask SmolLM2 to invent semantic answers in free-form English.

Instead:

    word
      ↓
    Graph-Topology memory retrieves ~32 candidate concepts
      ↓
    frozen LLM sees numbered candidates
      ↓
    LLM selects candidate NUMBERS only
      ↓
    program maps numbers back to concepts
      ↓
    compare against human semantic gold

The LLM is therefore a selector / language interface, not a semantic string
generator.

This avoids the V119 failure mode where the 360M model leaked prompt text and
invented junk concepts.

Conditions:
    GRAPH:
        candidates come from the learned V111 graph

    LEXICAL:
        candidates come from lexical-neighbor semantic structure

    GLOBAL:
        candidates come from globally common learned concepts

    LLM_ONLY:
        no candidates; tiny LLM emits free-form concepts (baseline only)

The main hypothesis is:

    GRAPH selection > LEXICAL selection > LLM_ONLY

The graph itself is NOT mutated during this run.

The semantic corpus is gold-only.

Default:
    512 words
    32 candidates
    max 4 selected concepts
    batch 128
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

MODEL_PATH = ROOT / "llm" / "SmolLM2-360M-Instruct"

MEMORY_PATH = ROOT / "results" / "v111_compact_semantic_memory.json"

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"

SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

OUTPUT_PATH = ROOT / "results" / "v120_constrained_graph_selection.json"

EVAL_WORDS = 512

BATCH_SIZE = 128

CANDIDATE_LIMIT = 32

MAX_SELECTED = 4

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 12

PRINT_EVERY = 128

TRACE_WORDS = {
    "hello",
    "greeting",
    "ability",
    "abandon",
    "water",
    "music",
    "animal",
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

    result = sorted(words)[:EVAL_WORDS]

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
# Human semantic gold
# ---------------------------------------------------------------------------

class HumanGold:
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
                            weight = (
                                frequency / n
                            )
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
# Learned V111 graph
# ---------------------------------------------------------------------------

class Memory:
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

        self.word_concepts = {
            str(word): [
                int(identifier)
                for identifier in ids
            ]
            for word, ids in payload.get(
                "word_concepts",
                {},
            ).items()
        }

        self.co_usage: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.rebuild_co_usage()

    def rebuild_co_usage(self) -> None:
        self.co_usage.clear()

        for identifiers in self.word_concepts.values():
            unique = list(
                dict.fromkeys(
                    identifiers
                )
            )

            for i, left in enumerate(unique):
                for right in unique[i + 1:]:
                    self.co_usage[
                        left
                    ][right] += 1

                    self.co_usage[
                        right
                    ][left] += 1

    def concepts_for_word(
        self,
        word: str,
    ) -> list[str]:
        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier
            in self.word_concepts.get(
                word,
                [],
            )
            if identifier
            in self.concept_name_by_id
        ]

    def top_used(
        self,
        limit: int,
    ) -> list[str]:
        ranked = sorted(
            self.usage.items(),
            key=lambda item: (
                -item[1],
                self.concept_name_by_id[
                    item[0]
                ],
            ),
        )

        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier, _count
            in ranked[:limit]
        ]

    def graph_candidates(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
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
                ] += 1.0

        for identifier, _score in scores.most_common(16):
            for related, count in self.co_usage.get(
                identifier,
                Counter(),
            ).most_common(8):
                scores[
                    related
                ] += (
                    0.25
                    * count
                )

        # Global graph usage as a weak fallback.
        for concept in self.top_used(16):
            identifier = self.concept_id_by_name.get(
                concept
            )

            if identifier is not None:
                scores[
                    identifier
                ] += 0.5

        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier, _score
            in scores.most_common(
                CANDIDATE_LIMIT
            )
        ]

    def lexical_candidates(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
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
                ] += 1.0

        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier, _score
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
                "Tokenizer has neither PAD nor EOS."
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
# Prompts
# ---------------------------------------------------------------------------

SELECT_SYSTEM = (
    "You are a semantic concept selector. "
    "Choose only from the numbered candidates. "
    "Return candidate numbers only."
)


def selection_prompt(
    tokenizer,
    word: str,
    candidates: list[str],
) -> str:
    numbered = "\n".join(
        f"{index + 1}. {concept}"
        for index, concept
        in enumerate(candidates)
    )

    return apply_chat(
        tokenizer,
        SELECT_SYSTEM,
        f"""
TARGET WORD:
{word}

CANDIDATE CONCEPTS:
{numbered}

Select up to {MAX_SELECTED} concepts that best describe the target.

Return ONLY space-separated candidate numbers.
Example:
2 7 13

Rules:
- Numbers must refer to the candidate list.
- Do not output words.
- Do not explain.
- If none fit, output 0.
""".strip(),
    )


def llm_only_prompt(
    tokenizer,
    word: str,
) -> str:
    return apply_chat(
        tokenizer,
        (
            "You are a semantic annotator. "
            "Return up to four simple concepts, comma-separated. "
            "No explanation."
        ),
        f"""
TARGET WORD:
{word}

Describe the word using simple reusable concepts.
Do not repeat the target word.
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

    for number in numbers:
        index = int(number)

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


def normalize_freeform(
    text: str,
    target: str,
) -> list[str]:
    result = []

    for part in re.split(
        r",|;|\n|\|",
        text,
    ):
        value = part.strip().lower()

        value = re.sub(
            r"^[\-\*\d\.\)\s]+",
            "",
            value,
        )

        value = re.sub(
            r"[^a-z0-9 \-]",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        if not value:
            continue

        if value == target:
            continue

        if len(value.split()) > 4:
            continue

        if len(value) > 40:
            continue

        if any(
            marker in value
            for marker in (
                "the answer",
                "the word",
                "assistant",
                "explanation",
            )
        ):
            continue

        if value not in result:
            result.append(
                value
            )

        if len(result) >= MAX_SELECTED:
            break

    return result


# ---------------------------------------------------------------------------
# Batched inference
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

    precision = (
        hits / len(predicted)
    )

    recall = (
        hits / len(gold)
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


def evaluate(
    predictions: dict[str, list[str]],
    gold: HumanGold,
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []

    for word, predicted in predictions.items():
        target = gold.gold(
            word
        )

        if not target:
            continue

        predicted_set = set(
            predicted
        )

        hits = len(
            predicted_set & target
        )

        precision = (
            hits
            / max(
                1,
                len(predicted_set),
            )
        )

        recall = (
            hits
            / max(
                1,
                len(target),
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
                predicted_set,
                target,
            )
        )

    return {
        "evaluated_words": len(f1s),
        "mean_precision": (
            sum(precisions)
            / max(
                1,
                len(precisions),
            )
        ),
        "mean_recall": (
            sum(recalls)
            / max(
                1,
                len(recalls),
            )
        ),
        "mean_f1": (
            sum(f1s)
            / max(
                1,
                len(f1s),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V120 CONSTRAINED GRAPH -> LLM SEMANTIC SELECTION ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    lexical_index = build_lexical_index(
        words
    )

    memory = Memory(
        MEMORY_PATH
    )

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    tokenizer, model, device = load_model()

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "graph_concepts:",
        len(memory.concept_id_by_name),
    )

    # ---------------------------------------------------------------
    # Prepare three constrained candidate conditions.
    # ---------------------------------------------------------------

    condition_builders = {
        "GRAPH": lambda word: memory.graph_candidates(
            word,
            lexical_index,
        ),
        "LEXICAL": lambda word: memory.lexical_candidates(
            word,
            lexical_index,
        ),
        "GLOBAL": lambda word: memory.top_used(
            CANDIDATE_LIMIT
        ),
    }

    results = {}

    for condition_name, candidate_builder in (
        condition_builders.items()
    ):
        print()
        print(
            f"=== CONDITION {condition_name} ===",
            flush=True,
        )

        predictions = {}
        invalid_output = 0
        total_selected = 0

        for start in range(
            0,
            len(words),
            BATCH_SIZE,
        ):
            batch_words = words[
                start:start + BATCH_SIZE
            ]

            batch_candidates = [
                candidate_builder(
                    word
                )
                for word in batch_words
            ]

            # Ensure every candidate list is non-empty where possible.
            batch_candidates = [
                candidates[
                    :CANDIDATE_LIMIT
                ]
                for candidates in batch_candidates
            ]

            prompts = [
                selection_prompt(
                    tokenizer,
                    word,
                    candidates,
                )
                for word, candidates
                in zip(
                    batch_words,
                    batch_candidates,
                )
            ]

            raw_outputs = generate_batch(
                tokenizer,
                model,
                device,
                prompts,
            )

            for (
                word,
                candidates,
                raw,
            ) in zip(
                batch_words,
                batch_candidates,
                raw_outputs,
            ):
                indices = parse_indices(
                    raw,
                    len(candidates),
                )

                if not indices:
                    invalid_output += 1

                concepts = [
                    candidates[index]
                    for index in indices
                ]

                predictions[word] = concepts
                total_selected += len(
                    concepts
                )

                if word in TRACE_WORDS:
                    print(
                        f"TRACE {condition_name} "
                        f"{word}: "
                        f"raw={raw!r} "
                        f"selected={concepts} "
                        f"gold={sorted(gold.gold(word))}",
                        flush=True,
                    )

            processed = min(
                start + BATCH_SIZE,
                len(words),
            )

            if (
                processed <= BATCH_SIZE
                or processed % PRINT_EVERY == 0
                or processed == len(words)
            ):
                print(
                    f"{condition_name} "
                    f"{processed:4d}/{len(words):4d} "
                    f"selected={total_selected} "
                    f"invalid={invalid_output}",
                    flush=True,
                )

        scores = evaluate(
            predictions,
            gold,
        )

        scores[
            "invalid_output"
        ] = invalid_output

        scores[
            "mean_selected"
        ] = (
            total_selected
            / max(
                1,
                len(words),
            )
        )

        results[
            condition_name
        ] = scores

    # ---------------------------------------------------------------
    # LLM-only baseline.
    # ---------------------------------------------------------------

    print()
    print(
        "=== CONDITION LLM_ONLY ===",
        flush=True,
    )

    llm_predictions = {}
    invalid_llm = 0
    total_selected_llm = 0

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        prompts = [
            llm_only_prompt(
                tokenizer,
                word,
            )
            for word in batch_words
        ]

        raw_outputs = generate_batch(
            tokenizer,
            model,
            device,
            prompts,
        )

        for (
            word,
            raw,
        ) in zip(
            batch_words,
            raw_outputs,
        ):
            concepts = normalize_freeform(
                raw,
                word,
            )

            if not concepts:
                invalid_llm += 1

            llm_predictions[
                word
            ] = concepts

            total_selected_llm += len(
                concepts
            )

        processed = min(
            start + BATCH_SIZE,
            len(words),
        )

        if (
            processed <= BATCH_SIZE
            or processed % PRINT_EVERY == 0
            or processed == len(words)
        ):
            print(
                f"LLM_ONLY "
                f"{processed:4d}/{len(words):4d} "
                f"selected={total_selected_llm} "
                f"invalid={invalid_llm}",
                flush=True,
            )

    llm_scores = evaluate(
        llm_predictions,
        gold,
    )

    llm_scores[
        "invalid_output"
    ] = invalid_llm

    llm_scores[
        "mean_selected"
    ] = (
        total_selected_llm
        / max(
            1,
            len(words),
        )
    )

    results[
        "LLM_ONLY"
    ] = llm_scores

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print()
    print(
        "=== V120 RESULTS ==="
    )

    print(
        "condition | F1 | precision | recall | selected | invalid"
    )

    for condition, result in results.items():
        print(
            f"{condition:9s} | "
            f"{result['mean_f1']:.4f} | "
            f"{result['mean_precision']:.4f} | "
            f"{result['mean_recall']:.4f} | "
            f"{result['mean_selected']:.2f} | "
            f"{result['invalid_output']}"
        )

    graph_f1 = results["GRAPH"]["mean_f1"]
    lexical_f1 = results["LEXICAL"]["mean_f1"]
    global_f1 = results["GLOBAL"]["mean_f1"]
    llm_f1 = results["LLM_ONLY"]["mean_f1"]

    print()
    print(
        "graph_minus_lexical_f1:",
        graph_f1 - lexical_f1,
    )

    print(
        "graph_minus_global_f1:",
        graph_f1 - global_f1,
    )

    print(
        "graph_minus_llm_only_f1:",
        graph_f1 - llm_f1,
    )

    report = {
        "experiment": (
            "V120 constrained graph -> LLM semantic selection"
        ),
        "evaluation_words": words,
        "candidate_limit": CANDIDATE_LIMIT,
        "max_selected": MAX_SELECTED,
        "results": results,
        "graph_minus_lexical_f1": (
            graph_f1 - lexical_f1
        ),
        "graph_minus_global_f1": (
            graph_f1 - global_f1
        ),
        "graph_minus_llm_only_f1": (
            graph_f1 - llm_f1
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V120 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
