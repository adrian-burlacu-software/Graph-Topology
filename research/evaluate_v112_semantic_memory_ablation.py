from __future__ import annotations

"""
V112 — SEMANTIC MEMORY ABLATION

Goal:
    Determine what the compressed graph/memory contributes beyond:
        A) human semantic retrieval
        B) lexical retrieval
        C) frozen LLM alone

Four conditions:

    A  HUMAN
       Human semantic corpus -> candidates -> frozen LLM

    B  GRAPH
       Accumulated compressed memory + lexical structure -> candidates
       -> frozen LLM

    C  LEXICAL
       Lexical neighbors only -> candidates -> frozen LLM

    D  LLM_ONLY
       No semantic candidates. The frozen LLM describes the word using
       a constrained simple-word prompt.

This is an evaluation experiment, NOT another training pass.

The experiment uses the 4925-word dictionary but only a sampled evaluation
set by default so it can finish quickly.

Important:
    The model remains frozen.
    The memory is loaded from the V111 JSON result if available.
    No giant vocabulary is placed into prompts.
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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

MEMORY_PATH = (
    ROOT / "results" / "v111_compact_semantic_memory.json"
)

OUTPUT_PATH = (
    ROOT / "results" / "v112_semantic_memory_ablation.json"
)

# Use 512 words for the overnight-friendly ablation.
# Set to None to evaluate all dictionary words.
EVAL_WORDS = 512

BATCH_SIZE = 128

CANDIDATE_LIMIT = 32

MAX_NEW_TOKENS = 24

MAX_INPUT_TOKENS = 256

MAX_CONCEPTS = 8


# ---------------------------------------------------------------------------
# Load dictionary
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

    result = sorted(words)

    if EVAL_WORDS is not None:
        result = result[:EVAL_WORDS]

    return result


# ---------------------------------------------------------------------------
# Human semantic corpus
# ---------------------------------------------------------------------------

class HumanSemanticIndex:
    def __init__(self) -> None:
        self.cue_features = defaultdict(Counter)
        self.feature_weight = Counter()

    def load(self, path: Path) -> None:
        started = time.perf_counter()

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
                            0,
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
                                0,
                            )
                        )

                        n = float(
                            row.get(
                                "n",
                                0,
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

                if weight <= 0:
                    continue

                self.cue_features[
                    cue
                ][feature] += weight

                self.feature_weight[
                    feature
                ] += weight

        print(
            "human_index_seconds:",
            f"{time.perf_counter() - started:.3f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Lexical index
# ---------------------------------------------------------------------------

def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    prefix = defaultdict(list)
    suffix = defaultdict(list)

    for word in words:
        prefix[
            word[:3]
        ].append(word)

        suffix[
            word[-3:]
        ].append(word)

    result = {}

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

        candidates.discard(word)

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

        result[word] = ranked[:12]

    return result


# ---------------------------------------------------------------------------
# Load V111 memory
# ---------------------------------------------------------------------------

class LoadedMemory:
    def __init__(
        self,
        path: Path,
    ) -> None:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        raw_ids = payload[
            "concept_id_by_name"
        ]

        self.concept_id_by_name = {
            str(name): int(identifier)
            for name, identifier
            in raw_ids.items()
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
            for word, ids
            in payload.get(
                "word_concepts",
                {},
            ).items()
        }

        self.learner_generated = set(
            payload.get(
                "learner_generated",
                [],
            )
        )

    def concepts_for_word(
        self,
        word: str,
    ) -> list[str]:
        return [
            self.concept_name_by_id[
                identifier
            ]
            for identifier in self.word_concepts.get(
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
            in ranked[:limit]
        ]


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def human_candidates(
    word: str,
    human: HumanSemanticIndex,
) -> list[str]:
    scores = Counter()

    for feature, weight in (
        human.cue_features.get(
            word,
            Counter(),
        ).items()
    ):
        scores[feature] += (
            3.0
            + math.log1p(
                max(
                    0,
                    weight,
                )
            )
        )

    for feature, weight in (
        human.feature_weight.most_common(
            16
        )
    ):
        scores[feature] += (
            0.5
            * math.log1p(
                max(
                    0,
                    weight,
                )
            )
        )

    return [
        feature
        for feature, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]


def lexical_candidates(
    word: str,
    human: HumanSemanticIndex,
    lexical: dict[str, list[str]],
) -> list[str]:
    scores = Counter()

    for neighbor in lexical.get(
        word,
        [],
    ):
        for feature, weight in (
            human.cue_features.get(
                neighbor,
                Counter(),
            ).most_common(8)
        ):
            scores[feature] += (
                1.0
                + 0.25
                * math.log1p(
                    max(
                        0,
                        weight,
                    )
                )
            )

    # Lexical-only fallback is literal lexical neighbors,
    # not human semantic features, if no semantic mapping exists.
    if not scores:
        return lexical.get(
            word,
            [],
        )[:CANDIDATE_LIMIT]

    return [
        feature
        for feature, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]


def graph_candidates(
    word: str,
    human: HumanSemanticIndex,
    lexical: dict[str, list[str]],
    memory: LoadedMemory,
) -> list[str]:
    scores = Counter()

    # Accumulated concept structure attached to lexical neighbors.
    for neighbor in lexical.get(
        word,
        [],
    ):
        for concept in memory.concepts_for_word(
            neighbor
        ):
            scores[concept] += 6.0

        for feature, weight in (
            human.cue_features.get(
                neighbor,
                Counter(),
            ).most_common(6)
        ):
            scores[feature] += (
                0.5
                + 0.1
                * math.log1p(
                    max(
                        0,
                        weight,
                    )
                )
            )

    # Global graph usage.
    for concept in memory.top_used(
        32
    ):
        scores[concept] += 2.0

    # Direct stored representation of this word is allowed for
    # the graph condition because it is precisely the accumulated graph.
    for concept in memory.concepts_for_word(
        word
    ):
        scores[concept] += 10.0

    if not scores:
        return human_candidates(
            word,
            human,
        )

    return [
        concept
        for concept, _score
        in scores.most_common(
            CANDIDATE_LIMIT
        )
    ]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a semantic annotator. "
    "Return only a comma-separated list of simple concepts. "
    "Never explain."
)


def candidate_prompt(
    tokenizer,
    word: str,
    candidates: list[str],
) -> str:
    user = f"""
TARGET: {word}

CANDIDATES:
{", ".join(candidates)}

Choose up to {MAX_CONCEPTS} concepts that describe the target.
Use only the supplied candidates.
Return one comma-separated line.
No explanation.
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM,
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
            SYSTEM
            + "\n\n"
            + user
            + "\n\nAssistant:"
        )


def llm_only_prompt(
    tokenizer,
    word: str,
) -> str:
    user = f"""
TARGET WORD: {word}

Describe this word using up to {MAX_CONCEPTS}
very simple English words.

Rules:
- use common simple words
- prefer concrete words when possible
- one word per concept
- comma-separated
- no explanation
- do not repeat the target word
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM,
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
            SYSTEM
            + "\n\n"
            + user
            + "\n\nAssistant:"
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(
    text: str,
    word: str,
    allowed: set[str] | None,
) -> list[str]:
    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.S,
    )

    parts = re.split(
        r",|;|\n|\|",
        text.lower(),
    )

    result = []
    seen = set()

    for raw in parts:
        item = raw.strip()

        item = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            item,
        )

        item = re.sub(
            r"^[\-\*]\s*",
            "",
            item,
        )

        item = re.sub(
            r"[^a-z0-9 \-]",
            " ",
            item,
        )

        item = re.sub(
            r"\s+",
            " ",
            item,
        ).strip()

        if not item:
            continue

        if item == word:
            continue

        if allowed is not None:
            if item not in allowed:
                continue

        if len(item) > 40:
            continue

        if item in seen:
            continue

        seen.add(item)
        result.append(item)

        if len(result) >= MAX_CONCEPTS:
            break

    return result


# ---------------------------------------------------------------------------
# Metrics against human corpus
# ---------------------------------------------------------------------------

def human_gold(
    word: str,
    human: HumanSemanticIndex,
) -> set[str]:
    return set(
        feature
        for feature, _weight
        in human.cue_features.get(
            word,
            Counter(),
        ).most_common(
            MAX_CONCEPTS
        )
    )


def score_prediction(
    predicted: list[str],
    gold: set[str],
) -> dict[str, float]:
    if not predicted:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    hits = len(
        set(predicted)
        & gold
    )

    precision = (
        hits
        / len(predicted)
    )

    recall = (
        hits
        / max(
            1,
            len(gold),
        )
    )

    if (
        precision
        + recall
        > 0
    ):
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
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

        generated = output[
            row,
            prompt_len:,
        ]

        results.append(
            tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()
        )

    return results


def run_condition(
    name: str,
    words: list[str],
    prompt_builder,
    tokenizer,
    model,
    device,
    gold_by_word: dict[str, set[str]],
) -> dict:
    print()
    print(
        f"=== CONDITION {name} ===",
        flush=True,
    )

    total = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    predicted_count = 0
    elapsed_start = time.perf_counter()

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch = words[
            start:start + BATCH_SIZE
        ]

        prompts = [
            prompt_builder(
                word
            )
            for word in batch
        ]

        raw = generate(
            tokenizer,
            model,
            device,
            prompts,
        )

        for word, text in zip(
            batch,
            raw,
        ):
            # Candidate conditions pass the candidate set through
            # the closure, while LLM_ONLY allows free output.
            allowed = None

            if hasattr(
                prompt_builder,
                "allowed_for_word",
            ):
                allowed = (
                    prompt_builder.allowed_for_word(
                        word
                    )
                )

            predicted = parse_output(
                text,
                word,
                allowed,
            )

            score = score_prediction(
                predicted,
                gold_by_word.get(
                    word,
                    set(),
                ),
            )

            for key in total:
                total[key] += score[key]

            predicted_count += len(
                predicted
            )

        processed = min(
            start + BATCH_SIZE,
            len(words),
        )

        if (
            processed <= BATCH_SIZE
            or processed % 128 == 0
            or processed == len(words)
        ):
            print(
                f"{name} "
                f"{processed}/{len(words)} "
                f"words/s="
                f"{processed / max(1e-9, time.perf_counter() - elapsed_start):.2f}",
                flush=True,
            )

    n = max(
        1,
        len(words),
    )

    seconds = (
        time.perf_counter()
        - elapsed_start
    )

    return {
        "words": len(words),
        "seconds": seconds,
        "words_per_second": (
            len(words)
            / max(
                1e-9,
                seconds,
            )
        ),
        "mean_precision": (
            total["precision"]
            / n
        ),
        "mean_recall": (
            total["recall"]
            / n
        ),
        "mean_f1": (
            total["f1"]
            / n
        ),
        "mean_predicted_concepts": (
            predicted_count
            / n
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V112 SEMANTIC MEMORY ABLATION ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    print(
        "evaluation_words:",
        len(words),
    )

    human = HumanSemanticIndex()
    human.load(
        SEMANTICS_PATH
    )

    lexical = build_lexical_index(
        words
    )

    memory = LoadedMemory(
        MEMORY_PATH
    )

    print(
        "loaded_memory_concepts:",
        len(
            memory.concept_id_by_name
        ),
    )

    gold_by_word = {
        word: human_gold(
            word,
            human,
        )
        for word in words
    }

    tokenizer, model, device = (
        load_model()
    )

    # ---------------------------------------------------------------
    # Candidate condition factories.
    # ---------------------------------------------------------------

    def make_candidate_condition(
        candidate_fn,
    ):
        def builder(word):
            candidates = candidate_fn(
                word
            )

            return candidate_prompt(
                tokenizer,
                word,
                candidates,
            )

        def allowed(word):
            return set(
                candidate_fn(
                    word
                )
            )

        builder.allowed_for_word = (
            allowed
        )

        return builder

    human_condition = (
        make_candidate_condition(
            lambda word:
                human_candidates(
                    word,
                    human,
                )
        )
    )

    lexical_condition = (
        make_candidate_condition(
            lambda word:
                lexical_candidates(
                    word,
                    human,
                    lexical,
                )
        )
    )

    graph_condition = (
        make_candidate_condition(
            lambda word:
                graph_candidates(
                    word,
                    human,
                    lexical,
                    memory,
                )
        )
    )

    def llm_builder(word):
        return llm_only_prompt(
            tokenizer,
            word,
        )

    # ---------------------------------------------------------------
    # Run all four.
    # ---------------------------------------------------------------

    results = {}

    results["human"] = run_condition(
        "HUMAN",
        words,
        human_condition,
        tokenizer,
        model,
        device,
        gold_by_word,
    )

    results["graph"] = run_condition(
        "GRAPH",
        words,
        graph_condition,
        tokenizer,
        model,
        device,
        gold_by_word,
    )

    results["lexical"] = run_condition(
        "LEXICAL",
        words,
        lexical_condition,
        tokenizer,
        model,
        device,
        gold_by_word,
    )

    results["llm_only"] = run_condition(
        "LLM_ONLY",
        words,
        llm_builder,
        tokenizer,
        model,
        device,
        gold_by_word,
    )

    print()
    print(
        "=== ABLATION RESULTS ==="
    )

    print(
        "condition | F1 | precision | recall | concepts/word | words/s"
    )

    for name, result in results.items():
        print(
            f"{name:9s} | "
            f"{result['mean_f1']:.4f} | "
            f"{result['mean_precision']:.4f} | "
            f"{result['mean_recall']:.4f} | "
            f"{result['mean_predicted_concepts']:.2f} | "
            f"{result['words_per_second']:.2f}"
        )

    # Pairwise deltas.
    graph = results["graph"]
    human_result = results["human"]
    lexical_result = results["lexical"]
    llm_result = results["llm_only"]

    print()
    print(
        "=== GRAPH CONTRIBUTION ==="
    )

    print(
        "graph_minus_lexical_f1:",
        graph["mean_f1"]
        - lexical_result["mean_f1"],
    )

    print(
        "graph_minus_llm_only_f1:",
        graph["mean_f1"]
        - llm_result["mean_f1"],
    )

    print(
        "graph_minus_human_f1:",
        graph["mean_f1"]
        - human_result["mean_f1"],
    )

    payload = {
        "experiment": "V112 semantic memory ablation",
        "evaluation_words": words,
        "batch_size": BATCH_SIZE,
        "candidate_limit": CANDIDATE_LIMIT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
        "graph_minus_lexical_f1": (
            graph["mean_f1"]
            - lexical_result["mean_f1"]
        ),
        "graph_minus_llm_only_f1": (
            graph["mean_f1"]
            - llm_result["mean_f1"]
        ),
        "graph_minus_human_f1": (
            graph["mean_f1"]
            - human_result["mean_f1"]
        ),
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
        "=== V112 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
