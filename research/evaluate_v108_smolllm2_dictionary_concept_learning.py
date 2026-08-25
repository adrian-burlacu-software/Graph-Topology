from __future__ import annotations

"""
V107 — FROZEN SmolLM2-360M-INSTRUCT -> COMPRESSED CONCEPT MEMORY

Goal
----
Start simple and then scale.

The frozen local LLM is asked to describe dictionary words using a small,
reusable vocabulary. A persistent Graph-Topology-style memory stores the
resulting concepts and grows incrementally.

Model:
    ./llm/SmolLM2-360M-Instruct

Input:
    ./data/dictionary.csv

Core loop:
    word
      ↓
    LLM description
      ↓
    normalized simple concepts
      ↓
    persistent memory
      ↓
    reuse existing concepts where possible
      ↓
    add only novel concepts

Two passes are used:

PASS 1 — TEACHER
    The LLM freely describes each word under strict constraints.

PASS 2 — REUSE
    The LLM is shown the memory vocabulary and asked to describe words using
    existing concepts whenever possible.

This gives us a direct first test of whether a frozen LLM can progressively
write into and then reuse a compressed external semantic memory.

Important:
    * LLM weights never change.
    * No hidden-state files are used.
    * No PT activation artifact is used.
    * The graph learns from textual interaction with the LLM.
    * The script is deterministic at temperature 0.
    * Memory is persistent within the run and saved at the end.

The experiment intentionally uses a compact output language.

Description format:
    concept, concept, concept, ...

Constraints:
    - maximum 10 concepts
    - no target word
    - simple English
    - avoid definitions / sentences
    - prefer reusable generic concepts
    - only lowercase ASCII-ish words/phrases
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
    / "v107_dictionary_concept_memory.json"
)

REPORT_PATH = (
    ROOT
    / "results"
    / "107.txt"
)

MAX_WORDS = None

# Start with a moderate cap so the first run remains manageable.
MAX_NEW_TOKENS = 64

# One short inference call per word per pass.
BATCH_SIZE = 1

MAX_CONCEPTS_PER_WORD = 10

# To keep the first pass cheap, start with N words. Set to None for all 4,925.
PASS1_WORD_LIMIT = None

# Pass 2 can be run over the whole corpus once the memory exists.
PASS2_WORD_LIMIT = None

# Memory vocabulary shown to the LLM is capped per prompt to avoid huge
# contexts. The complete memory remains on disk.
VOCAB_CONTEXT_LIMIT = 160

# Sampling is deterministic.
DO_SAMPLE = False


# ---------------------------------------------------------------------------
# Corpus
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
            "dictionary.csv produced no words."
        )

    return result


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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model directory: {MODEL_PATH}"
        )

    device = choose_device()

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
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def teacher_prompt(
    word: str,
) -> str:
    return f"""
Describe the English word "{word}" using simple reusable concepts.

Rules:
- Return ONLY a comma-separated list.
- Use at most {MAX_CONCEPTS_PER_WORD} concepts.
- Use simple English words or very short phrases.
- Do not use the target word.
- Do not write a sentence.
- Do not explain.
- Prefer concepts that could also describe other words.
- Prefer concrete properties, parts, uses, actions, states, or categories.

Example:
cat -> animal, pet, four legs, fur, tail

Now describe:
{word}
""".strip()


def reuse_prompt(
    word: str,
    vocabulary: list[str],
) -> str:
    vocab_text = ", ".join(
        vocabulary
    )

    return f"""
Describe the English word "{word}" using reusable concepts from the memory.

Known concepts:
{vocab_text}

Rules:
- Return ONLY a comma-separated list.
- Use at most {MAX_CONCEPTS_PER_WORD} concepts.
- Prefer concepts already in the known vocabulary.
- Introduce a new concept only when no known concept fits.
- Use simple English words or very short phrases.
- Do not use the target word.
- Do not write a sentence.
- Do not explain.

Now describe:
{word}
""".strip()


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    tokenizer,
    model,
    device,
    prompt: str,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    input_ids = inputs[
        "input_ids"
    ].to(device)

    attention_mask = inputs[
        "attention_mask"
    ].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = output[
        0,
        input_ids.shape[1]:,
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Concept normalization
# ---------------------------------------------------------------------------

def normalize_concept(
    value: str,
) -> Optional[str]:
    value = value.strip().lower()

    # Strip common markdown / numbering artifacts.
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

    # Keep short phrase-like units.
    if len(value) > 48:
        return None

    # Remove obvious sentence-like outputs.
    if any(
        marker in value
        for marker in (
            "\n",
            "?",
            " because ",
            " the word ",
            " means ",
            " is a ",
        )
    ):
        return None

    # Keep letters, spaces, hyphens.
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

    return value


def parse_concepts(
    text: str,
    target_word: str,
) -> list[str]:
    target = target_word.lower()

    # Remove common model framing.
    text = re.sub(
        r"^(answer|description|concepts?)\s*:\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    parts = re.split(
        r",|;|\n",
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

        # Avoid giant fragments.
        if len(concept.split()) > 4:
            continue

        if concept in seen:
            continue

        seen.add(concept)
        concepts.append(concept)

        if len(concepts) >= MAX_CONCEPTS_PER_WORD:
            break

    return concepts


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------

class ConceptMemory:
    """
    Minimal external semantic memory.

    concept_id:
        stable integer

    usage:
        number of words currently using the concept

    words:
        words that have been linked to the concept

    word_concepts:
        persistent word -> concept IDs

    The memory grows online; nothing is rebuilt when a new word arrives.
    """

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

        return (
            identifier,
            True,
        )

    def add_word(
        self,
        word: str,
        concepts: list[str],
    ) -> dict[str, int]:
        before_concepts = self.next_id
        new_links = 0
        reused_links = 0

        ids = []

        for concept in concepts:
            concept_id, created = (
                self.get_or_create(
                    concept
                )
            )

            if created:
                new_links += 1
            else:
                reused_links += 1

            ids.append(
                concept_id
            )

            self.usage[
                concept_id
            ] += 1

            self.words_by_concept[
                concept_id
            ].add(word)

        # Keep stable unique concept IDs.
        ids = list(
            dict.fromkeys(ids)
        )

        already = self.word_concepts.get(
            word
        )

        if already is None:
            self.word_concepts[
                word
            ] = ids
        else:
            self.word_concepts[
                word
            ] = list(
                dict.fromkeys(
                    already + ids
                )
            )

        return {
            "new_concepts": (
                self.next_id
                - before_concepts
            ),
            "new_links": new_links,
            "reused_links": reused_links,
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
            concept
            for concept, _identifier
            in ranked[:limit]
        ]

    def concept_stats(
        self,
    ) -> dict[str, float]:
        if not self.usage:
            return {
                "concepts": 0.0,
                "mean_usage": 0.0,
                "reused_fraction": 0.0,
            }

        counts = list(
            self.usage.values()
        )

        reused = sum(
            count > 1
            for count in counts
        )

        return {
            "concepts": float(
                len(self.concept_id_by_name)
            ),
            "mean_usage": (
                sum(counts)
                / len(counts)
            ),
            "reused_fraction": (
                reused
                / len(counts)
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
                str(key): value
                for key, value
                in self.usage.items()
            },
            "words_by_concept": {
                str(key): sorted(value)
                for key, value
                in self.words_by_concept.items()
            },
            "word_concepts": {
                key: value
                for key, value
                in self.word_concepts.items()
            },
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
# Metrics
# ---------------------------------------------------------------------------

def summarize_pass(
    stats: list[dict[str, float]],
    label: str,
) -> None:
    if not stats:
        return

    def avg(key: str) -> float:
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
        "mean_reused_links:",
        avg("reused_links"),
    )

    print(
        "mean_concepts_per_word:",
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
        "=== V107 FROZEN LLM -> COMPRESSED CONCEPT MEMORY ==="
    )
    print(
        "model:",
        MODEL_PATH,
    )
    print()

    words = load_dictionary(
        DICTIONARY_PATH
    )

    if PASS1_WORD_LIMIT is not None:
        pass1_words = words[
            :PASS1_WORD_LIMIT
        ]
    else:
        pass1_words = words

    if PASS2_WORD_LIMIT is not None:
        pass2_words = words[
            :PASS2_WORD_LIMIT
        ]
    else:
        pass2_words = words

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "pass1_words:",
        len(pass1_words),
    )

    print(
        "pass2_words:",
        len(pass2_words),
    )

    print()

    tokenizer, model, device = (
        load_model()
    )

    memory = ConceptMemory()

    # ---------------------------------------------------------------
    # PASS 1 — teacher descriptions
    # ---------------------------------------------------------------

    pass1_stats = []

    for index, word in enumerate(
        pass1_words,
        start=1,
    ):
        prompt = teacher_prompt(
            word
        )

        response = generate(
            tokenizer,
            model,
            device,
            prompt,
        )

        concepts = parse_concepts(
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

        pass1_stats.append(
            result
        )

        if (
            index <= 20
            or index % 100 == 0
            or index == len(pass1_words)
        ):
            print(
                f"PASS1 {index:4d}/{len(pass1_words):4d} "
                f"{word:20s} "
                f"concepts={concepts} "
                f"new={result['new_concepts']}",
                flush=True,
            )

    summarize_pass(
        pass1_stats,
        "V107 PASS1 TEACHER",
    )

    print(
        "memory_after_pass1:",
        memory.concept_stats(),
    )

    print()

    # ---------------------------------------------------------------
    # PASS 2 — reuse constrained by current vocabulary
    # ---------------------------------------------------------------

    pass2_stats = []

    for index, word in enumerate(
        pass2_words,
        start=1,
    ):
        vocabulary = memory.vocabulary(
            VOCAB_CONTEXT_LIMIT
        )

        prompt = reuse_prompt(
            word,
            vocabulary,
        )

        response = generate(
            tokenizer,
            model,
            device,
            prompt,
        )

        concepts = parse_concepts(
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

        pass2_stats.append(
            result
        )

        if (
            index <= 20
            or index % 100 == 0
            or index == len(pass2_words)
        ):
            print(
                f"PASS2 {index:4d}/{len(pass2_words):4d} "
                f"{word:20s} "
                f"concepts={concepts} "
                f"new={result['new_concepts']}",
                flush=True,
            )

    summarize_pass(
        pass2_stats,
        "V107 PASS2 REUSE",
    )

    final_stats = memory.concept_stats()

    print(
        "=== V107 FINAL MEMORY ==="
    )

    for key, value in final_stats.items():
        print(
            f"{key:24s}: {value}"
        )

    print()

    # Reuse concentration.
    ranked = sorted(
        memory.usage.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    print(
        "=== TOP REUSED CONCEPTS ==="
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
        "saved_memory:",
        OUTPUT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V107 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
