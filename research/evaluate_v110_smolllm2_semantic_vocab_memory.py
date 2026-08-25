from __future__ import annotations

"""
V109 — FROZEN SmolLM2-360M-INSTRUCT + HUMAN SEMANTIC VOCABULARY MEMORY

This is the cleaned-up version of V108.

The human semantic corpus is now the canonical concept vocabulary.

Inputs:
    ./llm/SmolLM2-360M-Instruct
    ./data/dictionary.csv
    ./data/semantics-large.csv

The semantic vocabulary is built from:
    translated
    weighted by normalized_translated

Architecture
------------
SEMANTICS-LARGE
    ↓
canonical human feature vocabulary
    ↓
seed external memory

DICTIONARY WORD
    ↓
frozen LLM
    ↓
select reusable human concepts
    ↓
graph stores concept IDs
    ↓
NEW:<concept> only when needed
    ↓
memory grows

We run TWO passes:

PASS 1
    The LLM selects from the human semantic vocabulary.

PASS 2
    The LLM sees the most-used concepts in the growing memory and is strongly
    encouraged to reuse them.

The important difference from V108:
    the model is no longer inventing the basic semantic vocabulary freely.

The persistent graph can therefore measure:
    * canonical concept reuse
    * novel concept creation
    * compression of word->concept mappings
    * concepts per word
    * coverage of the human semantic vocabulary
    * frequency-weighted reuse

No LLM weights are trained.
No hidden states are used.
No PT activation artifact is used.
"""

import csv
import json
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
    / "v109_semantic_vocab_memory.json"
)

BATCH_SIZE = 32
MAX_NEW_TOKENS = 48
MAX_INPUT_TOKENS = 768

MAX_CONCEPTS_PER_WORD = 8

# The full semantic corpus can contain thousands of distinct translated
# concepts. We expose a frequency-ranked working vocabulary to the small LLM.
SEED_VOCAB_SIZE = 512

# Memory vocabulary appended to the prompt in pass 2.
MEMORY_VOCAB_SIZE = 128

# Full dictionary.
MAX_WORDS = None

CHECKPOINT_EVERY = 250

DO_SAMPLE = False


# ---------------------------------------------------------------------------
# Corpus loading
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
            "dictionary.csv produced zero usable words."
        )

    return result


def load_semantic_vocabulary(
    path: Path,
) -> tuple[list[str], dict[str, float]]:
    """
    Build a canonical concept vocabulary from semantics-large.csv.

    Canonical concept:
        translated

    Weight:
        sum(normalized_translated)

    This retains the human-derived vocabulary while collapsing morphological
    variants such as:
        muscle
        muscles
        musculature
    into the canonical `muscle`.
    """
    weights = Counter()

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "translated",
            "normalized_translated",
            "frequency_translated",
            "n",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "semantics-large.csv missing: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            concept = row[
                "translated"
            ].strip().lower()

            if not concept:
                continue

            try:
                weight = float(
                    row[
                        "normalized_translated"
                    ]
                )
            except (
                TypeError,
                ValueError,
            ):
                weight = 0.0

            if weight <= 0.0:
                try:
                    frequency = float(
                        row[
                            "frequency_translated"
                        ]
                    )

                    n = int(
                        row["n"]
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    frequency = 0.0
                    n = 0

                if n > 0:
                    weight = (
                        frequency / n
                    )

            if weight > 0.0:
                weights[concept] += weight

    ranked = sorted(
        weights.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    vocabulary = [
        concept
        for concept, _weight
        in ranked[
            :SEED_VOCAB_SIZE
        ]
    ]

    return (
        vocabulary,
        dict(weights),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(
        torch.backends,
        "mps",
        None,
    ) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def load_model():
    device = choose_device()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        padding_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return (
        tokenizer,
        model,
        device,
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a compact semantic annotator.

Use the supplied human-derived semantic vocabulary.
Do not invent a general ontology.
Choose existing concepts whenever possible.

Your output is machine parsed.
Return ONLY:
    concept, concept, concept

or:
    NEW:concept, concept

No explanations.
No sentences.
""".strip()


def make_prompt(
    word: str,
    seed_vocabulary: list[str],
    memory_vocabulary: list[str],
) -> str:
    seed_text = ", ".join(
        seed_vocabulary
    )

    memory_text = ", ".join(
        memory_vocabulary
    )

    return f"""
Target word:
{word}

Human semantic vocabulary:
{seed_text}

Already-used memory concepts:
{memory_text if memory_text else "(none)"}

Rules:
1. Return at most {MAX_CONCEPTS_PER_WORD} items.
2. Each item must be either:
       an exact existing vocabulary concept
       OR NEW:<short concept>
3. Prefer existing concepts, especially already-used memory concepts.
4. A NEW concept should be used only when no existing concept is adequate.
5. Never include the target word.
6. Use exact vocabulary spelling when reusing a concept.
7. Return one comma-separated line only.
8. Do not explain anything.
""".strip()


def build_prompt(
    tokenizer,
    word: str,
    seed_vocabulary: list[str],
    memory_vocabulary: list[str],
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": make_prompt(
                word,
                seed_vocabulary,
                memory_vocabulary,
            ),
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
            + messages[1]["content"]
            + "\n\nAssistant:"
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_new_concept(
    text: str,
) -> Optional[str]:
    value = text.strip().lower()

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

    # Reject model meta-talk.
    bad = (
        "answer",
        "means",
        "description",
        "concepts",
        "word",
        "because",
        "the ",
        "a ",
        "an ",
    )

    if any(
        value == item
        or value.startswith(item)
        for item in bad
    ):
        return None

    return value


def parse_response(
    response: str,
    target_word: str,
    allowed: set[str],
) -> list[str]:
    target = target_word.lower()

    text = response.strip()

    text = re.sub(
        r"```(?:text|csv)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    )

    # Ignore obvious explanatory suffixes.
    for marker in (
        "\nExplanation:",
        "\nReason:",
        "\nWhy:",
    ):
        if marker in text:
            text = text.split(
                marker,
                1,
            )[0]

    parts = re.split(
        r",|;|\n|\|",
        text,
    )

    concepts = []
    seen = set()

    for raw in parts:
        item = raw.strip().lower()

        if not item:
            continue

        # Existing vocabulary concept.
        if (
            item in allowed
            and item != target
        ):
            if item not in seen:
                seen.add(item)
                concepts.append(item)

            continue

        # Explicit NEW: concept.
        if item.startswith(
            "new:"
        ):
            new_concept = normalize_new_concept(
                item
            )

            if (
                new_concept is not None
                and new_concept != target
                and new_concept not in seen
            ):
                seen.add(
                    "new:" + new_concept
                )
                concepts.append(
                    "NEW:" + new_concept
                )

        if (
            len(concepts)
            >= MAX_CONCEPTS_PER_WORD
        ):
            break

    return concepts


# ---------------------------------------------------------------------------
# Batch generation
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

        text = tokenizer.decode(
            output[
                row,
                prompt_len:,
            ],
            skip_special_tokens=True,
        ).strip()

        responses.append(
            text
        )

    return responses


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Graph-like external memory.

    The seed vocabulary comes from human semantic norms.

    New concepts may still appear, but they are marked as:
        learner_generated

    so we can distinguish human-vocabulary reuse from model expansion.
    """

    def __init__(
        self,
        seed_weights: dict[str, float],
    ) -> None:
        self.seed_weights = dict(
            seed_weights
        )

        self.seed_concepts = set(
            seed_weights
        )

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

        self.next_id = 0

        # Seed the graph with the human vocabulary.
        for concept in sorted(
            seed_weights
        ):
            self._create(
                concept
            )

    def _create(
        self,
        concept: str,
    ) -> int:
        identifier = self.next_id
        self.next_id += 1

        self.concept_id_by_name[
            concept
        ] = identifier

        self.concept_name_by_id[
            identifier
        ] = concept

        return identifier

    def get_or_create(
        self,
        concept: str,
    ) -> tuple[int, bool]:
        existing = self.concept_id_by_name.get(
            concept
        )

        if existing is not None:
            return (
                existing,
                False,
            )

        return (
            self._create(
                concept
            ),
            True,
        )

    def add_word(
        self,
        word: str,
        concepts: list[str],
    ) -> dict[str, int]:
        new_concepts = 0
        reused_concepts = 0
        seed_reuse = 0
        learner_new = 0

        ids = []

        for concept in concepts:
            is_new_marker = concept.startswith(
                "NEW:"
            )

            canonical = (
                concept[4:]
                if is_new_marker
                else concept
            )

            if not canonical:
                continue

            concept_id, created = (
                self.get_or_create(
                    canonical
                )
            )

            if created:
                new_concepts += 1
                learner_new += 1
            else:
                reused_concepts += 1

            if canonical in self.seed_concepts:
                seed_reuse += 1

            self.usage[
                concept_id
            ] += 1

            self.words_by_concept[
                concept_id
            ].add(word)

            ids.append(
                concept_id
            )

        self.word_concepts[
            word
        ] = list(
            dict.fromkeys(
                ids
            )
        )

        return {
            "new_concepts": new_concepts,
            "reused_concepts": reused_concepts,
            "seed_reuse": seed_reuse,
            "learner_new": learner_new,
        }

    def memory_vocabulary(
        self,
        limit: int = MEMORY_VOCAB_SIZE,
    ) -> list[str]:
        ranked = sorted(
            (
                (
                    name,
                    self.usage[
                        identifier
                    ],
                )
                for name, identifier
                in self.concept_id_by_name.items()
                if self.usage[
                    identifier
                ] > 0
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            name
            for name, _count
            in ranked[:limit]
        ]

    def stats(
        self,
    ) -> dict[str, float]:
        total_concepts = (
            len(
                self.concept_id_by_name
            )
        )

        learner_generated = (
            total_concepts
            - len(
                self.seed_concepts
            )
        )

        used_seed = sum(
            self.usage[
                self.concept_id_by_name[
                    concept
                ]
            ] > 0
            for concept
            in self.seed_concepts
            if concept
            in self.concept_id_by_name
        )

        used_total = sum(
            usage > 0
            for usage
            in self.usage.values()
        )

        return {
            "seed_vocab_size": float(
                len(
                    self.seed_concepts
                )
            ),
            "total_concepts": float(
                total_concepts
            ),
            "learner_generated_concepts": float(
                learner_generated
            ),
            "used_seed_concepts": float(
                used_seed
            ),
            "used_concepts": float(
                used_total
            ),
            "seed_coverage": (
                used_seed
                / max(
                    1,
                    len(
                        self.seed_concepts
                    ),
                )
            ),
            "mean_usage_per_used_concept": (
                sum(
                    usage
                    for usage
                    in self.usage.values()
                    if usage > 0
                )
                / max(
                    1,
                    used_total,
                )
            ),
        }

    def save(
        self,
        path: Path,
    ) -> None:
        payload = {
            "seed_concepts": sorted(
                self.seed_concepts
            ),
            "seed_weights": (
                self.seed_weights
            ),
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
                    words
                )
                for identifier, words
                in self.words_by_concept.items()
            },
            "word_concepts": (
                self.word_concepts
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
# Passes
# ---------------------------------------------------------------------------

def run_pass(
    label: str,
    words: list[str],
    tokenizer,
    model,
    device,
    memory: SemanticMemory,
    seed_vocabulary: list[str],
    use_memory: bool,
) -> list[dict[str, float]]:
    stats = []

    allowed = set(
        seed_vocabulary
    )

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        memory_vocabulary = (
            memory.memory_vocabulary(
                MEMORY_VOCAB_SIZE
            )
            if use_memory
            else []
        )

        prompts = [
            build_prompt(
                tokenizer,
                word,
                seed_vocabulary,
                memory_vocabulary,
            )
            for word in batch_words
        ]

        responses = generate_batch(
            tokenizer,
            model,
            device,
            prompts,
        )

        for word, response in zip(
            batch_words,
            responses,
        ):
            concepts = parse_response(
                response,
                word,
                allowed,
            )

            result = memory.add_word(
                word,
                concepts,
            )

            result[
                "concepts_returned"
            ] = len(
                concepts
            )

            stats.append(
                result
            )

        processed = min(
            start + BATCH_SIZE,
            len(words),
        )

        if (
            processed <= BATCH_SIZE * 2
            or processed % 100 == 0
            or processed == len(words)
        ):
            recent = stats[
                max(
                    0,
                    len(stats)
                    - BATCH_SIZE,
                ):
            ]

            def avg(
                key: str,
            ) -> float:
                return (
                    sum(
                        item[key]
                        for item
                        in recent
                    )
                    / max(
                        1,
                        len(recent),
                    )
                )

            print(
                f"{label} "
                f"{processed:4d}/{len(words):4d} "
                f"returned={avg('concepts_returned'):.2f} "
                f"seed_reuse={avg('seed_reuse'):.2f} "
                f"learner_new={avg('learner_new'):.2f}",
                flush=True,
            )

        if (
            processed % CHECKPOINT_EVERY == 0
            or processed == len(words)
        ):
            memory.save(
                OUTPUT_PATH
            )

    return stats


def summarize(
    stats: list[dict[str, float]],
    label: str,
) -> None:
    if not stats:
        return

    def avg(
        key: str,
    ) -> float:
        return sum(
            item[key]
            for item in stats
        ) / len(stats)

    print(
        f"=== {label} SUMMARY ==="
    )

    print(
        "words:",
        len(stats),
    )

    print(
        "mean_concepts_returned:",
        avg("concepts_returned"),
    )

    print(
        "mean_seed_reuse:",
        avg("seed_reuse"),
    )

    print(
        "mean_learner_new:",
        avg("learner_new"),
    )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V109 HUMAN SEMANTIC VOCABULARY MEMORY ==="
    )
    print(
        "Frozen model:",
        MODEL_PATH,
    )
    print()

    words = load_dictionary(
        DICTIONARY_PATH
    )

    seed_vocabulary, seed_weights = (
        load_semantic_vocabulary(
            SEMANTICS_PATH
        )
    )

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "human_seed_vocabulary:",
        len(seed_vocabulary),
    )

    print(
        "top_seed_concepts:",
        seed_vocabulary[:30],
    )

    print()

    tokenizer, model, device = (
        load_model()
    )

    memory = SemanticMemory(
        seed_weights
    )

    print(
        "initial_memory:",
        memory.stats(),
    )

    print()

    # ---------------------------------------------------------------
    # PASS 1 — canonical human vocabulary selection.
    # ---------------------------------------------------------------

    pass1 = run_pass(
        "PASS1",
        words,
        tokenizer,
        model,
        device,
        memory,
        seed_vocabulary,
        use_memory=False,
    )

    summarize(
        pass1,
        "V109 PASS1 CANONICAL VOCAB",
    )

    print(
        "after_pass1:",
        memory.stats(),
    )

    print()

    # ---------------------------------------------------------------
    # PASS 2 — memory-conditioned reuse.
    # ---------------------------------------------------------------

    pass2 = run_pass(
        "PASS2",
        words,
        tokenizer,
        model,
        device,
        memory,
        seed_vocabulary,
        use_memory=True,
    )

    summarize(
        pass2,
        "V109 PASS2 MEMORY REUSE",
    )

    print(
        "=== V109 FINAL MEMORY ==="
    )

    final_stats = memory.stats()

    for key, value in final_stats.items():
        print(
            f"{key:32s}: {value}"
        )

    print()

    print(
        "=== TOP USED HUMAN CONCEPTS ==="
    )

    ranked_seed = sorted(
        (
            (
                concept,
                memory.usage[
                    memory.concept_id_by_name[
                        concept
                    ]
                ],
            )
            for concept
            in memory.seed_concepts
            if concept
            in memory.concept_id_by_name
        ),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for concept, count in ranked_seed[:50]:
        print(
            f"{concept:32s} usage={count:5d}"
        )

    print()

    memory.save(
        OUTPUT_PATH
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
        "=== V109 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
