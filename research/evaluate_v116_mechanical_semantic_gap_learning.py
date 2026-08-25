from __future__ import annotations

"""
V116 — MECHANICAL SEMANTIC GAP LEARNING

V115 showed that SmolLM2-360M will happily invent slot answers rather than
emit GAP:. So V116 removes that burden from the model.

The model performs a constrained completion task, but the program itself
determines whether the graph can satisfy a slot.

For each word:

    graph context
        ↓
    ask LLM for a simple description using ONLY concepts from context
        ↓
    compare generated concepts against the allowed graph vocabulary
        ↓
    if coverage is insufficient:
        mechanically create a QUESTION
        ↓
    ask LLM to answer that concrete question
        ↓
    store existing concepts / NEW:<concept>
        ↓
    consolidate graph

The key difference:
    the LLM never has to say "I am uncertain".

A gap is an observable failure of the constrained task.

Two-stage task
--------------
1. RECONSTRUCT:
       "Describe TARGET using ONLY these concepts."

   If it produces too few valid concepts, that is a GAP.

2. QUESTION:
       "What concept is missing to explain TARGET in this context?"

   The answer may reuse existing concepts or create NEW:<concept>.

This is still a frozen LLM.
The V111 graph is the starting memory.
The semantic corpus is evaluation-only.

Full corpus:
    4925 words

Batch:
    128

Maximum rounds:
    2
"""

import csv
import json
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

MEMORY_INPUT_PATH = (
    ROOT
    / "results"
    / "v111_compact_semantic_memory.json"
)

MEMORY_OUTPUT_PATH = (
    ROOT
    / "results"
    / "v116_mechanical_gap_memory.json"
)

REPORT_OUTPUT_PATH = (
    ROOT
    / "results"
    / "v116_mechanical_gap_learning.json"
)

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"

SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

BATCH_SIZE = 128
ROUNDS = 2

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 24

CANDIDATE_LIMIT = 32
MAX_CONCEPTS = 8

# Minimum valid concepts required for the reconstruction task to count as
# successful. Keeping this small makes the gap detector tolerant of terse
# model answers.
MIN_VALID_CONCEPTS = 2

PRINT_EVERY = 128
CHECKPOINT_EVERY = 512

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

    result = sorted(words)

    if not result:
        raise RuntimeError(
            "dictionary.csv contained no usable words."
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
                                frequency
                                / n
                            )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        weight = 0.0

                if weight > 0.0:
                    self.cue_features[
                        cue
                    ][feature] += weight

    def gold(
        self,
        word: str,
        limit: int = 8,
    ) -> set[str]:
        return set(
            feature
            for feature, _weight
            in self.cue_features.get(
                word,
                Counter(),
            ).most_common(
                limit
            )
        )


# ---------------------------------------------------------------------------
# Learned graph
# ---------------------------------------------------------------------------

class LearnedMemory:
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

        self.word_concepts: dict[
            str,
            list[int],
        ] = {
            str(word): [
                int(identifier)
                for identifier
                in ids
            ]
            for word, ids
            in payload.get(
                "word_concepts",
                {},
            ).items()
        }

        self.words_by_concept: dict[
            int,
            set[str],
        ] = defaultdict(set)

        for word, identifiers in self.word_concepts.items():
            for identifier in identifiers:
                self.words_by_concept[
                    identifier
                ].add(word)

        self.learner_generated = set(
            payload.get(
                "learner_generated",
                [],
            )
        )

        self.co_usage: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        self.rebuild_co_usage()

    def rebuild_co_usage(
        self,
    ) -> None:
        self.co_usage.clear()

        for identifiers in self.word_concepts.values():
            unique = list(
                dict.fromkeys(
                    identifiers
                )
            )

            for i, left in enumerate(
                unique
            ):
                for right in unique[
                    i + 1:
                ]:
                    self.co_usage[
                        left
                    ][right] += 1

                    self.co_usage[
                        right
                    ][left] += 1

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

    def retrieve_context(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
        """
        Retrieve graph context with the target's own edges masked.
        """
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

        for identifier, _score in scores.most_common(
            16
        ):
            for related, count in self.co_usage.get(
                identifier,
                Counter(),
            ).most_common(
                8
            ):
                scores[
                    related
                ] += (
                    0.25
                    * count
                )

        target_ids = set(
            self.word_concepts.get(
                word,
                [],
            )
        )

        for identifier in target_ids:
            scores.pop(
                identifier,
                None,
            )

        for concept in self.top_used(
            16
        ):
            identifier = self.concept_id_by_name.get(
                concept
            )

            if (
                identifier is not None
                and identifier not in target_ids
            ):
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

    def add_concepts(
        self,
        word: str,
        concepts: list[str],
    ) -> int:
        created = 0

        ids = list(
            self.word_concepts.get(
                word,
                [],
            )
        )

        for raw in concepts:
            learner_new = raw.startswith(
                "NEW:"
            )

            concept = (
                raw[4:].strip().lower()
                if learner_new
                else raw.strip().lower()
            )

            concept = re.sub(
                r"\s+",
                " ",
                concept,
            )

            if not concept:
                continue

            identifier = (
                self.concept_id_by_name.get(
                    concept
                )
            )

            if identifier is None:
                identifier = len(
                    self.concept_id_by_name
                )

                self.concept_id_by_name[
                    concept
                ] = identifier

                self.concept_name_by_id[
                    identifier
                ] = concept

                created += 1

                if learner_new:
                    self.learner_generated.add(
                        concept
                    )

            if identifier not in ids:
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
        ] = ids

        return created

    def save(
        self,
        path: Path,
    ) -> None:
        payload = {
            "concept_id_by_name": self.concept_id_by_name,
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
            "word_concepts": self.word_concepts,
            "learner_generated": sorted(
                self.learner_generated
            ),
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
                "Tokenizer has neither PAD nor EOS token."
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

RECONSTRUCTION_SYSTEM = (
    "You are a semantic annotator. "
    "Use ONLY the supplied concept vocabulary. "
    "Do not invent concepts. "
    "Return only a comma-separated list."
)


def reconstruction_prompt(
    tokenizer,
    word: str,
    context: list[str],
) -> str:
    return apply_chat(
        tokenizer,
        RECONSTRUCTION_SYSTEM,
        f"""
TARGET:
{word}

AVAILABLE CONCEPTS:
{", ".join(context) if context else "(none)"}

Describe the target using up to {MAX_CONCEPTS} concepts.

Rules:
- You may ONLY use concepts from AVAILABLE CONCEPTS.
- Return only a comma-separated list.
- Do not explain.
- Do not use the target word.
""".strip(),
    )


QUESTION_SYSTEM = (
    "You identify missing semantic concepts for a memory. "
    "The candidate vocabulary is fixed. "
    "Return one short question only."
)


def question_prompt(
    tokenizer,
    word: str,
    context: list[str],
    attempted: list[str],
) -> str:
    return apply_chat(
        tokenizer,
        QUESTION_SYSTEM,
        f"""
TARGET:
{word}

AVAILABLE CONCEPTS:
{", ".join(context) if context else "(none)"}

CONCEPTS ALREADY TRIED:
{", ".join(attempted) if attempted else "(none)"}

The previous constrained description was insufficient.

Ask ONE short question whose answer would identify a reusable concept
that is missing from AVAILABLE CONCEPTS.

Good pattern:
"What kind of thing is this?"
"What is it used for?"
"What part or property distinguishes it?"

Return only the question.
""".strip(),
    )


ANSWER_SYSTEM = (
    "You answer semantic learning questions for an external memory. "
    "Reuse known concepts when possible. "
    "Use NEW:<concept> only when the missing concept is genuinely absent. "
    "Return only a comma-separated concept list."
)


def answer_prompt(
    tokenizer,
    word: str,
    question: str,
    context: list[str],
) -> str:
    return apply_chat(
        tokenizer,
        ANSWER_SYSTEM,
        f"""
TARGET:
{word}

QUESTION:
{question}

KNOWN CONCEPTS:
{", ".join(context) if context else "(none)"}

Return up to {MAX_CONCEPTS} concepts.

Rules:
- Existing concepts must match KNOWN CONCEPTS exactly.
- A genuinely missing concept may be NEW:<short concept>.
- No explanation.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize(
    value: str,
) -> str | None:
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

    if len(value) > 48:
        return None

    if len(value.split()) > 4:
        return None

    banned = (
        "the answer",
        "the word",
        "explanation",
        "because",
        "description",
    )

    if any(
        marker in value
        for marker in banned
    ):
        return None

    return value


def parse_existing_only(
    text: str,
    allowed: set[str],
    target: str,
) -> list[str]:
    concepts = []

    for part in re.split(
        r",|;|\n|\|",
        text,
    ):
        item = normalize(
            part
        )

        if (
            item is None
            or item == target
            or item not in allowed
        ):
            continue

        if item not in concepts:
            concepts.append(
                item
            )

        if len(concepts) >= MAX_CONCEPTS:
            break

    return concepts


def parse_question(
    text: str,
) -> str | None:
    text = text.strip()

    # Take the first sentence/question-like line.
    first = (
        text.splitlines()[0]
        if text
        else ""
    )

    first = first.strip(
        " \t"
    )

    if not first:
        return None

    if "?" in first:
        first = first[
            : first.find("?") + 1
        ]
    else:
        first += "?"

    if len(first) > 140:
        first = first[:140].rstrip() + "?"

    return first


def parse_answer(
    text: str,
    allowed: set[str],
    target: str,
) -> list[str]:
    concepts = []

    for part in re.split(
        r",|;|\n|\|",
        text,
    ):
        raw = part.strip()

        learner_new = raw.lower().startswith(
            "new:"
        )

        value = normalize(
            raw
        )

        if (
            value is None
            or value == target
        ):
            continue

        if learner_new:
            result = (
                "NEW:"
                + value
            )
        else:
            if value not in allowed:
                continue

            result = value

        if result not in concepts:
            concepts.append(
                result
            )

        if len(concepts) >= MAX_CONCEPTS:
            break

    return concepts


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


# ---------------------------------------------------------------------------
# Semantic evaluation
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


def semantic_f1(
    memory: LearnedMemory,
    gold: HumanGold,
    words: list[str],
) -> float:
    values = []

    for word in words:
        target = gold.gold(
            word
        )

        if not target:
            continue

        predicted = set(
            memory.concepts_for_word(
                word
            )
        )

        values.append(
            f1(
                predicted,
                target,
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
        "=== V116 MECHANICAL SEMANTIC GAP LEARNING ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    lexical_index = build_lexical_index(
        words
    )

    print(
        "dictionary_words:",
        len(words),
        flush=True,
    )

    print(
        "lexical_index_ready:",
        len(lexical_index),
        flush=True,
    )

    memory = LearnedMemory(
        MEMORY_INPUT_PATH
    )

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    tokenizer, model, device = (
        load_model()
    )

    initial_concepts = len(
        memory.concept_id_by_name
    )

    initial_f1 = semantic_f1(
        memory,
        gold,
        words,
    )

    print(
        "initial_concepts:",
        initial_concepts,
        flush=True,
    )

    print(
        "initial_semantic_f1:",
        initial_f1,
        flush=True,
    )

    total_gaps = 0
    total_questions = 0
    total_answers = 0
    total_created = 0

    round_stats = []
    trace = []

    for round_index in range(
        1,
        ROUNDS + 1,
    ):
        print()
        print(
            f"=== ROUND {round_index}/{ROUNDS} ===",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Phase A: constrained reconstruction.
        # ---------------------------------------------------------------

        gap_jobs = []

        for start in range(
            0,
            len(words),
            BATCH_SIZE,
        ):
            batch_words = words[
                start:start + BATCH_SIZE
            ]

            contexts = [
                memory.retrieve_context(
                    word,
                    lexical_index,
                )
                for word in batch_words
            ]

            prompts = [
                reconstruction_prompt(
                    tokenizer,
                    word,
                    context,
                )
                for word, context in zip(
                    batch_words,
                    contexts,
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
                context,
                raw,
            ) in zip(
                batch_words,
                contexts,
                raw_outputs,
            ):
                parsed = parse_existing_only(
                    raw,
                    set(context),
                    word,
                )

                # Mechanical failure condition.
                if len(parsed) < MIN_VALID_CONCEPTS:
                    total_gaps += 1

                    gap_jobs.append(
                        (
                            word,
                            context,
                            parsed,
                        )
                    )

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "phase": "RECONSTRUCT",
                            "round": round_index,
                            "word": word,
                            "context": context,
                            "raw": raw,
                            "parsed": parsed,
                            "gap": (
                                len(parsed)
                                < MIN_VALID_CONCEPTS
                            ),
                        }
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
                    f"RECON "
                    f"round={round_index} "
                    f"{processed:4d}/{len(words):4d} "
                    f"gaps={total_gaps} "
                    f"memory={len(memory.concept_id_by_name)}",
                    flush=True,
                )

        # ---------------------------------------------------------------
        # Phase B: generate one question for every mechanical gap.
        # ---------------------------------------------------------------

        question_jobs = []

        for start in range(
            0,
            len(gap_jobs),
            BATCH_SIZE,
        ):
            batch_jobs = gap_jobs[
                start:start + BATCH_SIZE
            ]

            prompts = [
                question_prompt(
                    tokenizer,
                    word,
                    context,
                    parsed,
                )
                for word, context, parsed
                in batch_jobs
            ]

            raw_questions = generate_batch(
                tokenizer,
                model,
                device,
                prompts,
            )

            for (
                (word, context, parsed),
                raw_question,
            ) in zip(
                batch_jobs,
                raw_questions,
            ):
                question = parse_question(
                    raw_question
                )

                if question is None:
                    continue

                total_questions += 1

                question_jobs.append(
                    (
                        word,
                        context,
                        question,
                    )
                )

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "phase": "QUESTION",
                            "round": round_index,
                            "word": word,
                            "context": context,
                            "question": question,
                        }
                    )

            processed = min(
                start + BATCH_SIZE,
                len(gap_jobs),
            )

            if (
                processed <= BATCH_SIZE
                or processed % PRINT_EVERY == 0
                or processed == len(gap_jobs)
            ):
                print(
                    f"QUESTION "
                    f"round={round_index} "
                    f"{processed:4d}/{len(gap_jobs):4d} "
                    f"questions={total_questions}",
                    flush=True,
                )

        # ---------------------------------------------------------------
        # Phase C: answer generated questions.
        # ---------------------------------------------------------------

        for start in range(
            0,
            len(question_jobs),
            BATCH_SIZE,
        ):
            batch_jobs = question_jobs[
                start:start + BATCH_SIZE
            ]

            prompts = [
                answer_prompt(
                    tokenizer,
                    word,
                    question,
                    context,
                )
                for word, context, question
                in batch_jobs
            ]

            raw_answers = generate_batch(
                tokenizer,
                model,
                device,
                prompts,
            )

            for (
                (word, context, question),
                raw_answer,
            ) in zip(
                batch_jobs,
                raw_answers,
            ):
                total_answers += 1

                allowed = set(
                    context
                )

                parsed = parse_answer(
                    raw_answer,
                    allowed,
                    word,
                )

                created = memory.add_concepts(
                    word,
                    parsed,
                )

                total_created += created

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "phase": "ANSWER",
                            "round": round_index,
                            "word": word,
                            "question": question,
                            "raw_answer": raw_answer,
                            "parsed": parsed,
                            "created": created,
                            "after": memory.concepts_for_word(
                                word
                            ),
                        }
                    )

            processed = min(
                start + BATCH_SIZE,
                len(question_jobs),
            )

            if (
                processed <= BATCH_SIZE
                or processed % PRINT_EVERY == 0
                or processed == len(question_jobs)
            ):
                print(
                    f"ANSWER "
                    f"round={round_index} "
                    f"{processed:4d}/{len(question_jobs):4d} "
                    f"new={total_created} "
                    f"memory={len(memory.concept_id_by_name)}",
                    flush=True,
                )

        memory.rebuild_co_usage()

        round_f1 = semantic_f1(
            memory,
            gold,
            words,
        )

        round_stats.append(
            {
                "round": round_index,
                "gaps": len(gap_jobs),
                "questions": len(question_jobs),
                "answers": len(question_jobs),
                "new_concepts": total_created,
                "memory_concepts": len(
                    memory.concept_id_by_name
                ),
                "semantic_f1": round_f1,
            }
        )

        print(
            f"ROUND {round_index} COMPLETE "
            f"gaps={len(gap_jobs)} "
            f"questions={len(question_jobs)} "
            f"new={total_created} "
            f"concepts={len(memory.concept_id_by_name)} "
            f"semantic_f1={round_f1:.4f}",
            flush=True,
        )

        memory.save(
            MEMORY_OUTPUT_PATH
        )

    final_f1 = semantic_f1(
        memory,
        gold,
        words,
    )

    print()
    print(
        "=== V116 SUMMARY ==="
    )

    print(
        "initial_concepts:",
        initial_concepts,
    )

    print(
        "final_concepts:",
        len(
            memory.concept_id_by_name
        ),
    )

    print(
        "concept_growth:",
        len(memory.concept_id_by_name)
        - initial_concepts,
    )

    print(
        "mechanical_gaps:",
        total_gaps,
    )

    print(
        "questions_generated:",
        total_questions,
    )

    print(
        "answers:",
        total_answers,
    )

    print(
        "new_concepts:",
        total_created,
    )

    print(
        "semantic_f1_before:",
        initial_f1,
    )

    print(
        "semantic_f1_after:",
        final_f1,
    )

    print(
        "semantic_f1_delta:",
        final_f1
        - initial_f1,
    )

    print()
    print(
        "=== TRACE ==="
    )

    for item in trace[:80]:
        print(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )

    report = {
        "experiment": (
            "V116 mechanical semantic gap learning"
        ),
        "initial_concepts": initial_concepts,
        "final_concepts": len(
            memory.concept_id_by_name
        ),
        "concept_growth": (
            len(memory.concept_id_by_name)
            - initial_concepts
        ),
        "mechanical_gaps": total_gaps,
        "questions_generated": total_questions,
        "answers": total_answers,
        "new_concepts": total_created,
        "semantic_f1_before": initial_f1,
        "semantic_f1_after": final_f1,
        "semantic_f1_delta": (
            final_f1
            - initial_f1
        ),
        "rounds": round_stats,
        "trace": trace,
        "word_concepts": memory.word_concepts,
        "concept_id_by_name": memory.concept_id_by_name,
        "usage": {
            str(identifier): count
            for identifier, count
            in memory.usage.items()
        },
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
    }

    REPORT_OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory.save(
        MEMORY_OUTPUT_PATH
    )

    print()
    print(
        "saved_memory:",
        MEMORY_OUTPUT_PATH,
    )

    print(
        "saved_report:",
        REPORT_OUTPUT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V116 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
