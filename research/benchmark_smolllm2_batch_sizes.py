from __future__ import annotations

"""
END-TO-END V110 BENCHMARK

Benchmarks the ACTUAL pipeline rather than generation alone.

Stages measured separately:

    1. semantic CSV indexing
    2. lexical candidate-index construction
    3. candidate retrieval for all dictionary words
    4. prompt construction
    5. tokenizer/batching
    6. GPU generation
    7. full end-to-end candidate retrieval + generation

This is intentionally a small representative run, so it is safe to execute
before committing to a full 4925-word run.

It also prints progress during every CPU stage so there is no silent
"nothing happening for two minutes" period.

Generation batch sizes:
    32, 64, 128

The end-to-end benchmark uses the same prompt shape as V110:
    ~32 candidates
    ~128 memory concepts

No semantic memory is mutated.
"""

import csv
import math
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

TEST_WORDS = 512

BATCH_SIZES = [32, 64, 128]

CANDIDATE_LIMIT = 32
MEMORY_VOCAB_LIMIT = 128

MAX_NEW_TOKENS = 24
MAX_INPUT_TOKENS = 384

# Number of retrieval rounds to benchmark separately.
RETRIEVAL_REPEATS = 2


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def now() -> float:
    return time.perf_counter()


def elapsed(
    start: float,
) -> float:
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def load_dictionary() -> list[str]:
    started = now()

    words = set()

    with DICTIONARY_PATH.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw in handle:
            word = raw.strip().lower()

            if word and word.isalpha():
                words.add(word)

    result = sorted(words)[:TEST_WORDS]

    print(
        f"[CPU] dictionary loaded: {len(result)} words "
        f"in {elapsed(started):.3f}s",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# Semantic index
# ---------------------------------------------------------------------------

class HumanSemanticIndex:
    def __init__(self) -> None:
        self.feature_weight = Counter()

        self.cue_features = defaultdict(
            Counter
        )

    def load(self, path: Path) -> None:
        started = now()

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                cue = row[
                    "cue"
                ].strip().lower()

                feature = row[
                    "translated"
                ].strip().lower()

                if not cue or not feature:
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
                        weight = frequency / n

                if weight <= 0.0:
                    continue

                self.cue_features[
                    cue
                ][feature] += weight

                self.feature_weight[
                    feature
                ] += weight

        print(
            f"[CPU] semantic index loaded: "
            f"{len(self.feature_weight)} concepts, "
            f"{len(self.cue_features)} cues "
            f"in {elapsed(started):.3f}s",
            flush=True,
        )

    def global_top(
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
# Fast lexical retrieval index
# ---------------------------------------------------------------------------

def build_fast_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    """
    Deliberately avoids Levenshtein.

    Build:
        prefix -> words
        suffix -> words

    Then produce a small deterministic neighborhood for each word.
    """
    started = now()

    prefix_buckets = defaultdict(list)
    suffix_buckets = defaultdict(list)

    for word in words:
        prefix_buckets[
            word[:3]
        ].append(word)

        suffix_buckets[
            word[-3:]
        ].append(word)

    neighbors = {}

    for index, word in enumerate(words):
        candidates = set()

        candidates.update(
            prefix_buckets[
                word[:3]
            ]
        )

        candidates.update(
            suffix_buckets[
                word[-3:]
            ]
        )

        candidates.discard(word)

        # No edit-distance calculation here.
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

        neighbors[word] = ranked[:12]

        if (
            index < 5
            or (index + 1) % 100 == 0
            or index + 1 == len(words)
        ):
            print(
                f"[CPU] lexical index "
                f"{index + 1:4d}/{len(words):4d}",
                flush=True,
            )

    print(
        f"[CPU] lexical index complete in "
        f"{elapsed(started):.3f}s",
        flush=True,
    )

    return neighbors


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(
    word: str,
    semantic_index: HumanSemanticIndex,
    lexical_neighbors: dict[str, list[str]],
    memory_concepts: list[str],
) -> list[str]:
    scores = Counter()

    # Direct human cue associations.
    for feature, weight in (
        semantic_index.cue_features.get(
            word,
            Counter(),
        ).items()
    ):
        scores[feature] += (
            2.0
            + math.log1p(
                max(
                    0.0,
                    weight,
                )
            )
        )

    # Neighbor cue associations.
    for neighbor in lexical_neighbors.get(
        word,
        [],
    ):
        for feature, weight in (
            semantic_index.cue_features.get(
                neighbor,
                Counter(),
            ).most_common(8)
        ):
            scores[feature] += (
                1.0
                + 0.25
                * math.log1p(
                    max(
                        0.0,
                        weight,
                    )
                )
            )

    # Current memory vocabulary gets a substantial preference.
    for concept in memory_concepts:
        scores[concept] += 4.0

    # Global fallback.
    for feature, weight in (
        semantic_index.feature_weight.most_common(
            16
        )
    ):
        scores[feature] += (
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

    result = [
        feature
        for feature, _score
        in ranked[
            :CANDIDATE_LIMIT
        ]
    ]

    return result


# ---------------------------------------------------------------------------
# Prompting
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
    memory_concepts: list[str],
) -> str:
    user = f"""
TARGET: {word}

CANDIDATES:
{", ".join(candidates)}

ALREADY REUSED:
{", ".join(memory_concepts) if memory_concepts else "(none)"}

Choose up to 8 concepts.
Prefer exact candidate concepts.
Prefer already reused concepts when they fit.
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


def build_all_prompts(
    tokenizer,
    words: list[str],
    semantic_index: HumanSemanticIndex,
    lexical_neighbors: dict[str, list[str]],
    memory_concepts: list[str],
) -> list[str]:
    started = now()

    prompts = []

    for index, word in enumerate(words):
        candidates = retrieve_candidates(
            word,
            semantic_index,
            lexical_neighbors,
            memory_concepts,
        )

        prompts.append(
            build_prompt(
                tokenizer,
                word,
                candidates,
                memory_concepts,
            )
        )

        if (
            index < 5
            or (index + 1) % 100 == 0
            or index + 1 == len(words)
        ):
            print(
                f"[CPU] prompts "
                f"{index + 1:4d}/{len(words):4d}",
                flush=True,
            )

    print(
        f"[CPU] prompt/retrieval complete in "
        f"{elapsed(started):.3f}s",
        flush=True,
    )

    return prompts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "loading model...",
        flush=True,
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
                "Tokenizer has no pad/eos token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    print(
        "model ready:",
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_prompts(
    tokenizer,
    model,
    device,
    prompts: list[str],
    batch_size: int,
) -> dict[str, float]:
    started = now()

    total_input_tokens = 0
    total_output_tokens = 0
    total_words = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(
            device
        )

    for start in range(
        0,
        len(prompts),
        batch_size,
    ):
        batch = prompts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
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

        total_input_tokens += int(
            attention_mask.sum().item()
        )

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

        # Count actual newly generated positions row by row.
        for row in range(
            output.shape[0]
        ):
            prompt_len = int(
                attention_mask[row].sum().item()
            )

            total_output_tokens += max(
                0,
                output.shape[1]
                - prompt_len,
            )

        total_words += len(batch)

        if (
            total_words <= batch_size * 2
            or total_words % 128 == 0
            or total_words == len(prompts)
        ):
            print(
                f"[GPU] generated "
                f"{total_words:4d}/{len(prompts):4d} "
                f"(batch={batch_size})",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

        peak_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (
                1024 * 1024
            )
        )
    else:
        peak_mb = 0.0

    seconds = elapsed(
        started
    )

    return {
        "seconds": seconds,
        "words": float(
            len(prompts)
        ),
        "words_per_second": (
            len(prompts)
            / max(
                1e-9,
                seconds,
            )
        ),
        "input_tokens": float(
            total_input_tokens
        ),
        "output_tokens": float(
            total_output_tokens
        ),
        "output_tokens_per_second": (
            total_output_tokens
            / max(
                1e-9,
                seconds,
            )
        ),
        "mean_input_tokens": (
            total_input_tokens
            / max(
                1,
                len(prompts),
            )
        ),
        "mean_output_tokens": (
            total_output_tokens
            / max(
                1,
                len(prompts),
            )
        ),
        "peak_gpu_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(
        "=== END-TO-END V110 BENCHMARK ==="
    )
    print(
        f"test_words={TEST_WORDS}"
    )
    print(
        f"candidate_limit={CANDIDATE_LIMIT}"
    )
    print(
        f"memory_vocab_limit={MEMORY_VOCAB_LIMIT}"
    )
    print(
        f"max_new_tokens={MAX_NEW_TOKENS}"
    )
    print()

    words = load_dictionary()

    semantic_index = (
        HumanSemanticIndex()
    )

    semantic_index.load(
        SEMANTICS_PATH
    )

    lexical_neighbors = (
        build_fast_lexical_index(
            words
        )
    )

    # Representative memory vocabulary.
    memory_concepts = (
        semantic_index.global_top(
            MEMORY_VOCAB_LIMIT
        )
    )

    tokenizer, model, device = (
        load_model()
    )

    # ---------------------------------------------------------------
    # CPU candidate/prompt stage
    # ---------------------------------------------------------------

    prompts = build_all_prompts(
        tokenizer,
        words,
        semantic_index,
        lexical_neighbors,
        memory_concepts,
    )

    # Tokenization-only measurement.
    started = now()

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    )

    tokenization_seconds = elapsed(
        started
    )

    mean_input = (
        sum(
            len(row)
            for row in tokenized[
                "input_ids"
            ]
        )
        / max(
            1,
            len(prompts),
        )
    )

    print()
    print(
        "=== CPU STAGE SUMMARY ==="
    )

    print(
        "prompt_build_seconds:",
        "see progress timing above",
    )

    print(
        "tokenization_seconds:",
        f"{tokenization_seconds:.3f}",
    )

    print(
        "mean_input_tokens:",
        mean_input,
    )

    print()

    # ---------------------------------------------------------------
    # GPU generation tests
    # ---------------------------------------------------------------

    results = []

    for batch_size in BATCH_SIZES:
        print()
        print(
            f"=== GENERATION batch={batch_size} ==="
        )

        try:
            # Small warmup.
            warmup_count = min(
                batch_size * 2,
                len(prompts),
            )

            generate_prompts(
                tokenizer,
                model,
                device,
                prompts[
                    :warmup_count
                ],
                batch_size,
            )

            result = generate_prompts(
                tokenizer,
                model,
                device,
                prompts,
                batch_size,
            )

            result[
                "batch_size"
            ] = batch_size

            results.append(
                result
            )

            print(
                f"batch={batch_size} "
                f"seconds={result['seconds']:.3f} "
                f"words/s={result['words_per_second']:.2f} "
                f"out_tok/s={result['output_tokens_per_second']:.2f} "
                f"peak_MB={result['peak_gpu_mb']:.1f}",
                flush=True,
            )

        except torch.cuda.OutOfMemoryError:
            print(
                f"batch={batch_size} -> CUDA OOM",
                flush=True,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print()
    print(
        "=== FINAL RESULTS ==="
    )

    print(
        "batch | seconds | words/s | out_tok/s | peak_MB | in_tok | out_tok"
    )

    for result in results:
        print(
            f"{int(result['batch_size']):5d} | "
            f"{result['seconds']:8.3f} | "
            f"{result['words_per_second']:7.2f} | "
            f"{result['output_tokens_per_second']:10.2f} | "
            f"{result['peak_gpu_mb']:8.1f} | "
            f"{result['mean_input_tokens']:6.1f} | "
            f"{result['mean_output_tokens']:7.1f}"
        )

    if results:
        fastest = max(
            results,
            key=lambda item: (
                item[
                    "words_per_second"
                ]
            ),
        )

        print()
        print(
            "=== RECOMMENDED ==="
        )
        print(
            "batch_size:",
            int(
                fastest["batch_size"]
            ),
        )
        print(
            "words_per_second:",
            fastest[
                "words_per_second"
            ],
        )

    print()
    print(
        "=== END-TO-END BENCH COMPLETE ==="
    )


if __name__ == "__main__":
    main()
