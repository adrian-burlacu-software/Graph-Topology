from __future__ import annotations

"""
V108 — BATCHED FROZEN SmolLM2-360M-INSTRUCT -> CONCEPT MEMORY

Fixes V107's two main problems:

1. SPEED
   V107 did one generate() call per word.
   V108 batches many prompts into one GPU call.

2. OUTPUT DISCIPLINE
   V108 uses the tokenizer's chat template when available and makes the
   model emit a compact machine-readable concept line.

Architecture:

    dictionary word
         |
         v
    frozen SmolLM2-Instruct
         |
         v
    simple reusable concept list
         |
         v
    persistent concept memory

Pass 1:
    describe each word independently.

Pass 2:
    present the current memory vocabulary and force reuse where possible.

No LLM weights are trained.
No embedding/PT artifact is used.
"""

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
# Paths / config
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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v108_dictionary_concept_memory.json"
)

# Batch generation is the major speedup.
BATCH_SIZE = 32

MAX_NEW_TOKENS = 40
MAX_INPUT_TOKENS = 768

MAX_CONCEPTS_PER_WORD = 8

# Full corpus by default.
MAX_WORDS = None

# Prompt vocabulary cap for pass 2.
VOCAB_CONTEXT_LIMIT = 128

# Checkpoint every N processed words.
CHECKPOINT_EVERY = 250

# Greedy decoding.
DO_SAMPLE = False


# ---------------------------------------------------------------------------
# Device/model
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
            "dictionary.csv contained no usable words."
        )

    return result


def load_model():
    device = choose_device()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    print(
        "model:",
        MODEL_PATH,
    )
    print(
        "device:",
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
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
                "Tokenizer has no pad/eos token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You describe English words using a tiny reusable concept vocabulary. "
    "Return only concepts, not explanations."
)


def make_user_prompt(
    word: str,
    vocabulary: Optional[list[str]] = None,
) -> str:
    if vocabulary:
        vocab = ", ".join(
            vocabulary
        )

        return (
            "Describe the word "
            f'"{word}"'
            " using concepts from this memory vocabulary whenever possible.\n"
            f"MEMORY: {vocab}\n\n"
            f"Rules: return only 1 to {MAX_CONCEPTS_PER_WORD} short concepts "
            "separated by commas. Prefer known concepts. "
            "Introduce a new concept only when needed. "
            "Do not include the target word. "
            "Do not write a sentence. "
            "No labels. No explanation."
        )

    return (
        "Describe the word "
        f'"{word}"'
        " using simple reusable concepts.\n\n"
        f"Rules: return only 1 to {MAX_CONCEPTS_PER_WORD} short concepts "
        "separated by commas. "
        "Use simple English. "
        "Prefer properties, categories, parts, actions, uses, or states. "
        "Do not include the target word. "
        "Do not write a sentence. "
        "No labels. No explanation."
    )


def build_prompt(
    tokenizer,
    word: str,
    vocabulary: Optional[list[str]] = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": make_user_prompt(
                word,
                vocabulary,
            ),
        },
    ]

    if hasattr(
        tokenizer,
        "apply_chat_template",
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback for tokenizers without a usable chat template.
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + messages[1]["content"]
        + "\n\nAssistant:"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_concept(
    text: str,
) -> Optional[str]:
    value = text.strip().lower()

    value = re.sub(
        r"^[\-\*\d\.\)\s]+",
        "",
        value,
    )

    value = value.strip(
        " \t\r\n.,;:!?\"'`()[]{}"
    )

    if not value:
        return None

    # Reject obvious meta-language.
    meta = (
        "answer",
        "the answer",
        "word means",
        "word that means",
        "description",
        "concepts:",
        "here are",
        "such as",
    )

    if any(
        marker in value
        for marker in meta
    ):
        return None

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

    if len(value) > 48:
        return None

    if len(value.split()) > 4:
        return None

    return value


def parse_response(
    response: str,
    target_word: str,
) -> list[str]:
    """
    The model is expected to produce:
        concept, concept, concept

    We stop at the first obvious sentence/meta separator.
    """
    target = target_word.lower()

    text = response.strip()

    # Strip markdown code fences.
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

    # Ignore anything after a sentence-style explanation.
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

    for part in parts:
        concept = normalize_concept(
            part
        )

        if concept is None:
            continue

        if concept == target:
            continue

        if concept in seen:
            continue

        seen.add(concept)
        concepts.append(concept)

        if (
            len(concepts)
            >= MAX_CONCEPTS_PER_WORD
        ):
            break

    return concepts


# ---------------------------------------------------------------------------
# Batch inference
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

        generated_ids = output[
            row,
            prompt_len:,
        ]

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        responses.append(
            text
        )

    return responses


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

class ConceptMemory:
    def __init__(self) -> None:
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

    def get_or_create(
        self,
        concept: str,
    ) -> tuple[int, bool]:
        existing = self.concept_id_by_name.get(
            concept
        )

        if existing is not None:
            return existing, False

        identifier = self.next_id
        self.next_id += 1

        self.concept_id_by_name[
            concept
        ] = identifier

        self.concept_name_by_id[
            identifier
        ] = concept

        return identifier, True

    def add_word(
        self,
        word: str,
        concepts: list[str],
    ) -> dict[str, int]:
        new_concepts = 0
        reused_concepts = 0

        ids = []

        for concept in concepts:
            concept_id, created = (
                self.get_or_create(
                    concept
                )
            )

            if created:
                new_concepts += 1
            else:
                reused_concepts += 1

            ids.append(
                concept_id
            )

            self.usage[
                concept_id
            ] += 1

            self.words_by_concept[
                concept_id
            ].add(word)

        ids = list(
            dict.fromkeys(
                ids
            )
        )

        self.word_concepts[
            word
        ] = ids

        return {
            "new_concepts": new_concepts,
            "reused_concepts": reused_concepts,
        }

    def vocabulary(
        self,
        limit: int = VOCAB_CONTEXT_LIMIT,
    ) -> list[str]:
        ranked = sorted(
            self.concept_id_by_name.items(),
            key=lambda item: (
                -self.usage[
                    item[1]
                ],
                item[0],
            ),
        )

        return [
            name
            for name, _identifier
            in ranked[:limit]
        ]

    def stats(
        self,
    ) -> dict[str, float]:
        if not self.usage:
            return {
                "concepts": 0.0,
                "mean_usage": 0.0,
                "reused_fraction": 0.0,
            }

        usages = list(
            self.usage.values()
        )

        return {
            "concepts": float(
                len(
                    self.concept_id_by_name
                )
            ),
            "mean_usage": (
                sum(usages)
                / len(usages)
            ),
            "reused_fraction": (
                sum(
                    usage > 1
                    for usage in usages
                )
                / len(usages)
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
            "concept_name_by_id": (
                self.concept_name_by_id
            ),
            "usage": {
                str(key): value
                for key, value
                in self.usage.items()
            },
            "words_by_concept": {
                str(key): sorted(value)
                for key, value
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
# Pass execution
# ---------------------------------------------------------------------------

def run_pass(
    label: str,
    words: list[str],
    tokenizer,
    model,
    device,
    memory: ConceptMemory,
    use_memory: bool,
) -> list[dict[str, float]]:
    stats = []

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        vocabulary = (
            memory.vocabulary(
                VOCAB_CONTEXT_LIMIT
            )
            if use_memory
            else None
        )

        prompts = [
            build_prompt(
                tokenizer,
                word,
                vocabulary,
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
            )

            result = memory.add_word(
                word,
                concepts,
            )

            result[
                "concepts_returned"
            ] = len(concepts)

            stats.append(
                {
                    **result,
                    "concepts_returned": len(
                        concepts
                    ),
                }
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
                    len(stats) - BATCH_SIZE,
                ):
            ]

            mean_new = (
                sum(
                    item["new_concepts"]
                    for item in recent
                )
                / max(
                    1,
                    len(recent),
                )
            )

            print(
                f"{label} "
                f"{processed:4d}/{len(words):4d} "
                f"mean_new_last_batch={mean_new:.2f} "
                f"memory_concepts={len(memory.concept_id_by_name)}",
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
            float(item[key])
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
        "mean_new_concepts:",
        avg("new_concepts"),
    )

    print(
        "mean_reused_concepts:",
        avg("reused_concepts"),
    )

    print(
        "mean_concepts_returned:",
        avg("concepts_returned"),
    )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== V108 BATCHED SmolLM2-360M-INSTRUCT -> CONCEPT MEMORY ==="
    )
    print(
        "model:",
        MODEL_PATH,
    )
    print(
        "batch_size:",
        BATCH_SIZE,
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    print(
        "dictionary_words:",
        len(words),
    )

    print()

    tokenizer, model, device = (
        load_model()
    )

    memory = ConceptMemory()

    # ---------------------------------------------------------------
    # PASS 1
    # ---------------------------------------------------------------

    print(
        "=== V108 PASS 1: TEACHER ==="
    )

    pass1_stats = run_pass(
        "PASS1",
        words,
        tokenizer,
        model,
        device,
        memory,
        use_memory=False,
    )

    summarize(
        pass1_stats,
        "V108 PASS 1",
    )

    print(
        "memory_after_pass1:",
        memory.stats(),
    )

    print()

    # ---------------------------------------------------------------
    # PASS 2
    # ---------------------------------------------------------------

    print(
        "=== V108 PASS 2: MEMORY REUSE ==="
    )

    pass2_stats = run_pass(
        "PASS2",
        words,
        tokenizer,
        model,
        device,
        memory,
        use_memory=True,
    )

    summarize(
        pass2_stats,
        "V108 PASS 2",
    )

    print(
        "memory_after_pass2:",
        memory.stats(),
    )

    print()

    print(
        "=== V108 TOP REUSED CONCEPTS ==="
    )

    ranked = sorted(
        memory.usage.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for concept_id, count in ranked[:50]:
        print(
            f"{memory.concept_name_by_id[concept_id]:32s} "
            f"usage={count:4d}"
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
        "=== V108 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
