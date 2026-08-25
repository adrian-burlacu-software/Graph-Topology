from __future__ import annotations

"""
V111 — FROZEN SmolLM2 -> COMPACT SEMANTIC CONCEPT MEMORY

Full 4925-word experiment.

Architecture:
    semantics-large.csv
          |
          v
    human semantic vocabulary
          |
          +----> fast candidate retrieval (~32 concepts)
          |
    dictionary word
          |
          v
    frozen SmolLM2-360M-Instruct
          |
          v
    choose existing concepts / NEW:<concept>
          |
          v
    persistent semantic memory
          |
          +----> influences later candidate retrieval

Important:
    The full semantic vocabulary is NEVER placed in the LLM prompt.
    The memory vocabulary is NEVER dumped into the LLM prompt.
    Only ~32 retrieved candidates are shown.

Two passes:
    PASS 1: build the memory from scratch.
    PASS 2: revisit the same 4925 words and test whether the accumulated
            structure causes more reuse / fewer new concepts.

No model training.
No .pt activation file.
No embeddings required.

Designed for:
    RTX 4060 Laptop GPU
    SmolLM2-360M-Instruct
"""

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v111_compact_semantic_memory.json"
)

BATCH_SIZE = 128

CANDIDATE_LIMIT = 32

MAX_CONCEPTS_PER_WORD = 8

MAX_NEW_TOKENS = 24

MAX_INPUT_TOKENS = 256

CHECKPOINT_EVERY = 500

PRINT_EVERY = 128

# Keep this small: memory affects retrieval, not prompt size.
MEMORY_RETRIEVAL_LIMIT = 32

DO_SAMPLE = False


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

    if not result:
        raise RuntimeError(
            f"No dictionary words found in {path}"
        )

    return result


# ---------------------------------------------------------------------------
# Human semantic corpus
# ---------------------------------------------------------------------------

def safe_float(
    value,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


class HumanSemanticIndex:
    def __init__(self) -> None:
        self.feature_weight: Counter[str] = Counter()

        self.cue_features: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)

        self.feature_cues: dict[
            str,
            set[str],
        ] = defaultdict(set)

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

            fields = set(
                reader.fieldnames or []
            )

            missing = required - fields

            if missing:
                raise RuntimeError(
                    "semantics-large.csv is missing: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

            rows = 0

            for row in reader:
                rows += 1

                cue = row[
                    "cue"
                ].strip().lower()

                feature = row[
                    "translated"
                ].strip().lower()

                if not cue or not feature:
                    continue

                weight = safe_float(
                    row[
                        "normalized_translated"
                    ]
                )

                if weight <= 0:
                    frequency = safe_float(
                        row[
                            "frequency_translated"
                        ]
                    )

                    n = safe_float(
                        row["n"]
                    )

                    if n > 0:
                        weight = (
                            frequency / n
                        )

                if weight <= 0:
                    continue

                self.cue_features[
                    cue
                ][feature] += weight

                self.feature_weight[
                    feature
                ] += weight

                self.feature_cues[
                    feature
                ].add(cue)

        seconds = (
            time.perf_counter()
            - started
        )

        print(
            f"semantic_rows={rows} "
            f"cues={len(self.cue_features)} "
            f"concepts={len(self.feature_weight)} "
            f"seconds={seconds:.3f}",
            flush=True,
        )

    def top_global(
        self,
        limit: int,
    ) -> list[str]:
        return [
            feature
            for feature, _weight
            in self.feature_weight.most_common(
                limit
            )
        ]


# ---------------------------------------------------------------------------
# Fast lexical retrieval
#
# IMPORTANT:
# No Levenshtein / pairwise distance.
# ---------------------------------------------------------------------------

def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    started = time.perf_counter()

    prefix: dict[
        str,
        list[str],
    ] = defaultdict(list)

    suffix: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for word in words:
        prefix[
            word[:3]
        ].append(word)

        suffix[
            word[-3:]
        ].append(word)

    result: dict[
        str,
        list[str],
    ] = {}

    for word in words:
        candidates = set()

        candidates.update(
            prefix[
                word[:3]
            ]
        )

        candidates.update(
            suffix[
                word[-3:]
            ]
        )

        candidates.discard(
            word
        )

        ranked = sorted(
            candidates,
            key=lambda other: (
                not other.startswith(
                    word[:3]
                ),
                not other.endswith(
                    word[-3:]
                ),
                abs(
                    len(other)
                    - len(word)
                ),
                other,
            ),
        )

        result[
            word
        ] = ranked[:12]

    print(
        f"lexical_index_words={len(result)} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class SemanticMemory:
    def __init__(
        self,
        human: HumanSemanticIndex,
    ) -> None:
        self.human = human

        self.concept_id_by_name: dict[
            str,
            int,
        ] = {}

        self.concept_name_by_id: dict[
            int,
            str,
        ] = {}

        self.usage: Counter[int] = Counter()

        self.words_by_concept: dict[
            int,
            set[str],
        ] = defaultdict(set)

        self.word_concepts: dict[
            str,
            list[int],
        ] = {}

        self.learner_generated: set[
            str
        ] = set()

        self.next_id = 0

    def get_or_create(
        self,
        concept: str,
        learner_created: bool,
    ) -> tuple[int, bool]:
        existing = (
            self.concept_id_by_name.get(
                concept
            )
        )

        if existing is not None:
            return (
                existing,
                False,
            )

        identifier = self.next_id

        self.next_id += 1

        self.concept_id_by_name[
            concept
        ] = identifier

        self.concept_name_by_id[
            identifier
        ] = concept

        if learner_created:
            self.learner_generated.add(
                concept
            )

        return (
            identifier,
            True,
        )

    def add_word(
        self,
        word: str,
        concepts: list[str],
    ) -> dict[str, int]:
        reused = 0
        human_reused = 0
        learner_new = 0
        created_total = 0

        ids = []

        for raw in concepts:
            learner_created = raw.startswith(
                "NEW:"
            )

            concept = (
                raw[4:].strip()
                if learner_created
                else raw.strip()
            )

            concept = re.sub(
                r"\s+",
                " ",
                concept.lower(),
            )

            if not concept:
                continue

            identifier, created = (
                self.get_or_create(
                    concept,
                    learner_created,
                )
            )

            if created:
                created_total += 1

            else:
                reused += 1

                if (
                    concept
                    in self.human.feature_weight
                ):
                    human_reused += 1

            if learner_created:
                learner_new += 1

            ids.append(
                identifier
            )

            self.usage[
                identifier
            ] += 1

            self.words_by_concept[
                identifier
            ].add(word)

        self.word_concepts[
            word
        ] = list(
            dict.fromkeys(
                ids
            )
        )

        return {
            "created_total": created_total,
            "reused": reused,
            "human_reused": human_reused,
            "learner_new": learner_new,
            "returned": len(
                ids
            ),
        }

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
            in ranked[
                :limit
            ]
        ]

    def stats(
        self,
    ) -> dict[str, float]:
        used = [
            count
            for count in self.usage.values()
            if count > 0
        ]

        used_human = 0
        used_learner = 0

        for concept, identifier in (
            self.concept_id_by_name.items()
        ):
            if self.usage[
                identifier
            ] <= 0:
                continue

            if concept in (
                self.human.feature_weight
            ):
                used_human += 1
            else:
                used_learner += 1

        return {
            "total_concepts": float(
                len(
                    self.concept_id_by_name
                )
            ),
            "used_concepts": float(
                len(used)
            ),
            "used_human_concepts": float(
                used_human
            ),
            "used_learner_concepts": float(
                used_learner
            ),
            "learner_generated_concepts": float(
                len(
                    self.learner_generated
                )
            ),
            "total_assignments": float(
                sum(used)
            ),
            "mean_usage_per_used_concept": (
                sum(used)
                / max(
                    1,
                    len(used),
                )
            ),
        }

    def save(
        self,
        path: Path,
    ) -> None:
        payload = {
            "concept_id_by_name": (
                self.concept_id_by_name
            ),
            "usage": {
                str(identifier): count
                for identifier, count
                in self.usage.items()
            },
            "words_by_concept": {
                str(identifier): sorted(
                    values
                )
                for identifier, values
                in self.words_by_concept.items()
            },
            "word_concepts": (
                self.word_concepts
            ),
            "learner_generated": sorted(
                self.learner_generated
            ),
            "next_id": self.next_id,
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
    human: HumanSemanticIndex,
    lexical: dict[str, list[str]],
    memory: SemanticMemory,
) -> list[str]:
    scores = Counter()

    # Direct semantic corpus evidence.
    for feature, weight in (
        human.cue_features.get(
            word,
            Counter(),
        ).items()
    ):
        scores[
            feature
        ] += (
            3.0
            + math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

    # Evidence from lexical neighbors.
    for neighbor in lexical.get(
        word,
        [],
    ):
        for feature, weight in (
            human.cue_features.get(
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

        # Existing graph structure gets a bonus.
        for concept in memory.concepts_for_word(
            neighbor
        ):
            scores[
                concept
            ] += 5.0

    # Previously successful concepts globally.
    for concept in memory.top_used(
        MEMORY_RETRIEVAL_LIMIT
    ):
        scores[
            concept
        ] += 2.0

    # Global human fallback.
    for feature, weight in (
        human.feature_weight.most_common(
            16
        )
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

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        concept
        for concept, _score
        in ranked[
            :CANDIDATE_LIMIT
        ]
    ]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a semantic annotator. "
    "Choose concepts from the supplied candidate vocabulary. "
    "Return only a comma-separated list. "
    "Never explain."
)


def build_prompt(
    tokenizer,
    word: str,
    candidates: list[str],
) -> str:
    user = f"""
TARGET: {word}

CANDIDATES:
{", ".join(candidates)}

Choose up to {MAX_CONCEPTS_PER_WORD} concepts.
Prefer exact candidate concepts.
If an important concept is missing, you may write NEW:<short concept>.
Never use the target word.
Return one comma-separated line.
No explanation.
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
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
            SYSTEM_PROMPT
            + "\n\n"
            + user
            + "\n\nAssistant:"
        )


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

    print(
        "model:",
        MODEL_PATH,
        flush=True,
    )

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

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither PAD nor EOS token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
    )

    model.eval()
    model.to(device)

    return (
        tokenizer,
        model,
        device,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def normalize_new(
    value: str,
) -> Optional[str]:
    value = value.strip().lower()

    value = re.sub(
        r"^new\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
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
        return None

    if len(value) > 40:
        return None

    if len(value.split()) > 4:
        return None

    banned = (
        "the answer",
        "a word that",
        "means",
        "the word",
        "concepts",
        "description",
    )

    if any(
        item in value
        for item in banned
    ):
        return None

    return value


def parse_response(
    text: str,
    word: str,
    candidates: set[str],
) -> list[str]:
    text = text.strip()

    text = re.sub(
        r"```[a-zA-Z]*",
        "",
        text,
    )

    text = text.replace(
        "```",
        "",
    )

    parts = re.split(
        r",|;|\n|\|",
        text,
    )

    result = []
    seen = set()

    for part in parts:
        item = part.strip().lower()

        if not item:
            continue

        if (
            item in candidates
            and item != word
        ):
            if item not in seen:
                seen.add(item)
                result.append(item)

        elif item.startswith(
            "new:"
        ):
            concept = normalize_new(
                item
            )

            if (
                concept
                and concept != word
            ):
                marker = (
                    "NEW:"
                    + concept
                )

                if marker not in seen:
                    seen.add(marker)
                    result.append(marker)

        if len(result) >= (
            MAX_CONCEPTS_PER_WORD
        ):
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
        do_sample=DO_SAMPLE,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    responses = []

    for row in range(
        output.shape[0]
    ):
        prompt_len = int(
            attention_mask[row].sum().item()
        )

        generated = output[
            row,
            prompt_len:,
        ]

        responses.append(
            tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
        )

    return responses


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------

def run_pass(
    pass_name: str,
    words: list[str],
    human: HumanSemanticIndex,
    lexical: dict[str, list[str]],
    memory: SemanticMemory,
    tokenizer,
    model,
    device,
) -> list[dict[str, int]]:
    print()
    print(
        f"=== {pass_name} ===",
        flush=True,
    )

    results = []

    started = time.perf_counter()

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        retrievals = []

        prompts = []

        for word in batch_words:
            candidates = (
                retrieve_candidates(
                    word,
                    human,
                    lexical,
                    memory,
                )
            )

            retrievals.append(
                candidates
            )

            prompts.append(
                build_prompt(
                    tokenizer,
                    word,
                    candidates,
                )
            )

        responses = generate_batch(
            tokenizer,
            model,
            device,
            prompts,
        )

        for word, candidates, response in zip(
            batch_words,
            retrievals,
            responses,
        ):
            parsed = parse_response(
                response,
                word,
                set(candidates),
            )

            result = memory.add_word(
                word,
                parsed,
            )

            result[
                "candidate_count"
            ] = len(candidates)

            result[
                "word"
            ] = word

            results.append(
                result
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
            recent = results[
                max(
                    0,
                    len(results) - BATCH_SIZE,
                ):
            ]

            returned = (
                sum(
                    row[
                        "returned"
                    ]
                    for row in recent
                )
                / max(
                    1,
                    len(recent),
                )
            )

            reused = (
                sum(
                    row[
                        "reused"
                    ]
                    for row in recent
                )
                / max(
                    1,
                    len(recent),
                )
            )

            human_reused = (
                sum(
                    row[
                        "human_reused"
                    ]
                    for row in recent
                )
                / max(
                    1,
                    len(recent),
                )
            )

            learner_new = (
                sum(
                    row[
                        "learner_new"
                    ]
                    for row in recent
                )
                / max(
                    1,
                    len(recent),
                )
            )

            rate = (
                processed
                / max(
                    1e-9,
                    time.perf_counter()
                    - started,
                )
            )

            print(
                f"{pass_name} "
                f"{processed:4d}/{len(words):4d} "
                f"returned={returned:.2f} "
                f"reused={reused:.2f} "
                f"human_reuse={human_reused:.2f} "
                f"learner_new={learner_new:.2f} "
                f"memory={len(memory.concept_id_by_name)} "
                f"words/s={rate:.2f}",
                flush=True,
            )

        if (
            processed % CHECKPOINT_EVERY == 0
            or processed == len(words)
        ):
            memory.save(
                OUTPUT_PATH
            )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mean(
    rows: list[dict[str, int]],
    key: str,
) -> float:
    return (
        sum(
            row[key]
            for row in rows
        )
        / max(
            1,
            len(rows),
        )
    )


def print_pass_summary(
    name: str,
    rows: list[dict[str, int]],
) -> None:
    print()
    print(
        f"=== {name} SUMMARY ==="
    )

    print(
        "mean_candidates:",
        mean(
            rows,
            "candidate_count",
        ),
    )

    print(
        "mean_returned:",
        mean(
            rows,
            "returned",
        ),
    )

    print(
        "mean_reused:",
        mean(
            rows,
            "reused",
        ),
    )

    print(
        "mean_human_reused:",
        mean(
            rows,
            "human_reused",
        ),
    )

    print(
        "mean_learner_new:",
        mean(
            rows,
            "learner_new",
        ),
    )

    print(
        "total_learner_new:",
        sum(
            row[
                "learner_new"
            ]
            for row in rows
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V111 COMPACT FROZEN LLM -> SEMANTIC MEMORY ==="
    )

    print(
        "model:",
        MODEL_PATH,
    )

    print(
        "dictionary:",
        DICTIONARY_PATH,
    )

    print(
        "semantics:",
        SEMANTICS_PATH,
    )

    print(
        "batch_size:",
        BATCH_SIZE,
    )

    print(
        "candidate_limit:",
        CANDIDATE_LIMIT,
    )

    print(
        "max_new_tokens:",
        MAX_NEW_TOKENS,
    )

    print()

    words = load_dictionary(
        DICTIONARY_PATH
    )

    print(
        "dictionary_words:",
        len(words),
        flush=True,
    )

    human = HumanSemanticIndex()

    human.load(
        SEMANTICS_PATH
    )

    print(
        "human_seed_concepts:",
        len(
            human.feature_weight
        ),
        flush=True,
    )

    lexical = build_lexical_index(
        words
    )

    tokenizer, model, device = (
        load_model()
    )

    memory = SemanticMemory(
        human
    )

    print()
    print(
        "initial_memory:",
        memory.stats(),
        flush=True,
    )

    pass1 = run_pass(
        "PASS1",
        words,
        human,
        lexical,
        memory,
        tokenizer,
        model,
        device,
    )

    print_pass_summary(
        "PASS1",
        pass1,
    )

    print(
        "memory_after_pass1:",
        memory.stats(),
        flush=True,
    )

    pass2 = run_pass(
        "PASS2",
        words,
        human,
        lexical,
        memory,
        tokenizer,
        model,
        device,
    )

    print_pass_summary(
        "PASS2",
        pass2,
    )

    print()
    print(
        "=== FINAL ==="
    )

    print(
        "memory:",
        memory.stats(),
    )

    print(
        "total_words_processed:",
        len(words) * 2,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print()
    print(
        "=== TOP USED CONCEPTS ==="
    )

    ranked = sorted(
        memory.usage.items(),
        key=lambda item: (
            -item[1],
            memory.concept_name_by_id[
                item[0]
            ],
        ),
    )

    for identifier, count in ranked[:50]:
        concept = (
            memory.concept_name_by_id[
                identifier
            ]
        )

        print(
            f"{concept:32s} usage={count:5d}"
        )

    memory.save(
        OUTPUT_PATH
    )

    print()
    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "=== V111 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
