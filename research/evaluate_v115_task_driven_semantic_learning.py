from __future__ import annotations

"""
V115 — TASK-DRIVEN SEMANTIC LEARNING

This replaces V114's failed "are you uncertain?" metacognition test.

Instead of asking a frozen 360M model to decide whether it knows enough, we
give it a concrete semantic task with explicit required slots.

Task:
    describe a word using four roles:

        CATEGORY
        PROPERTY
        USE
        RELATION

For every slot the model must return either:
    an existing concept
    or
    GAP:<short question>

A GAP is therefore produced by failure to complete a required task, not by
asking the model to introspect about uncertainty.

Then:

    GAP
      ↓
    LLM answers the generated question
      ↓
    existing concepts / NEW:<concept>
      ↓
    graph consolidation
      ↓
    task repeated

The graph is the only source of memory during learning.

semantics-large.csv is evaluation-only.

The run starts from V111 memory, writes a new V115 memory file, and never
modifies V111.

This is a full-corpus experiment:
    4925 words
    2 task/inquiry rounds
    batch size 128

No hidden-state inference.
No PT activation artifact.
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
MEMORY_INPUT_PATH = ROOT / "results" / "v111_compact_semantic_memory.json"
MEMORY_OUTPUT_PATH = ROOT / "results" / "v115_task_driven_memory.json"

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

BATCH_SIZE = 128
ROUNDS = 2

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 32

CANDIDATE_LIMIT = 32
MAX_CONCEPTS_PER_ANSWER = 8

PRINT_EVERY = 128
CHECKPOINT_EVERY = 512


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

    result = sorted(words)

    if not result:
        raise RuntimeError("dictionary.csv contained no words")

    return result


def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    prefix: dict[str, list[str]] = defaultdict(list)
    suffix: dict[str, list[str]] = defaultdict(list)

    for word in words:
        prefix[word[:3]].append(word)
        suffix[word[-3:]].append(word)

    result: dict[str, list[str]] = {}

    for word in words:
        candidates = set(prefix[word[:3]])
        candidates.update(suffix[word[-3:]])
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
# Evaluation-only human semantic gold
# ---------------------------------------------------------------------------

class HumanGold:
    def __init__(self) -> None:
        self.cue_features: dict[str, Counter[str]] = defaultdict(Counter)

    def load(self, path: Path) -> None:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
            errors="replace",
        ) as handle:
            reader = csv.DictReader(handle)

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
                        if n > 0.0:
                            weight = frequency / n
                    except (TypeError, ValueError):
                        weight = 0.0

                if weight > 0.0:
                    self.cue_features[cue][feature] += weight

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
# Learned graph memory
# ---------------------------------------------------------------------------

class LearnedMemory:
    def __init__(self, path: Path) -> None:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.concept_id_by_name = {
            str(name): int(identifier)
            for name, identifier
            in payload["concept_id_by_name"].items()
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
                in payload.get("usage", {}).items()
            }
        )

        self.word_concepts: dict[str, list[int]] = {
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

        self.words_by_concept: dict[int, set[str]] = defaultdict(set)

        for word, identifiers in self.word_concepts.items():
            for identifier in identifiers:
                self.words_by_concept[identifier].add(word)

        self.learner_generated = set(
            payload.get(
                "learner_generated",
                [],
            )
        )

        self.co_usage: dict[int, Counter[int]] = defaultdict(Counter)
        self.rebuild_co_usage()

    def rebuild_co_usage(self) -> None:
        self.co_usage.clear()

        for identifiers in self.word_concepts.values():
            unique = list(dict.fromkeys(identifiers))

            for i, left in enumerate(unique):
                for right in unique[i + 1:]:
                    self.co_usage[left][right] += 1
                    self.co_usage[right][left] += 1

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
            learner_new = raw.startswith("NEW:")

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

            identifier = self.concept_id_by_name.get(
                concept
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
                ids.append(identifier)

            self.usage[identifier] += 1
            self.words_by_concept[identifier].add(word)

        self.word_concepts[word] = ids

        return created

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
            if identifier in self.concept_name_by_id
        ]

    def top_used(
        self,
        limit: int,
    ) -> list[str]:
        ranked = sorted(
            self.usage.items(),
            key=lambda item: (
                -item[1],
                self.concept_name_by_id[item[0]],
            ),
        )

        return [
            self.concept_name_by_id[identifier]
            for identifier, _count
            in ranked[:limit]
        ]

    def retrieve_context(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
        """
        Graph-only context with the target word's own edges masked.
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
                scores[identifier] += 1.0

        for identifier, _score in scores.most_common(16):
            for related, count in self.co_usage.get(
                identifier,
                Counter(),
            ).most_common(8):
                scores[related] += 0.25 * count

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

        for concept in self.top_used(16):
            identifier = self.concept_id_by_name.get(
                concept
            )

            if (
                identifier is not None
                and identifier not in target_ids
            ):
                scores[identifier] += 0.5

        return [
            self.concept_name_by_id[identifier]
            for identifier, _score
            in scores.most_common(
                CANDIDATE_LIMIT
            )
        ]

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
                str(identifier): sorted(words)
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
# Task prompts
# ---------------------------------------------------------------------------

TASK_SYSTEM = (
    "You are a semantic learner using an external concept memory. "
    "Complete the requested semantic task. "
    "For every required slot, output an existing concept or GAP:<question>. "
    "Return only the requested lines."
)


def task_prompt(
    tokenizer,
    word: str,
    context: list[str],
) -> str:
    context_text = (
        ", ".join(context)
        if context
        else "(none)"
    )

    return apply_chat(
        tokenizer,
        TASK_SYSTEM,
        f"""
TARGET WORD:
{word}

CURRENT GRAPH CONTEXT:
{context_text}

Complete these FOUR slots:

CATEGORY=
PROPERTY=
USE=
RELATION=

For each slot write exactly one of:
    an existing concept from CURRENT GRAPH CONTEXT
    GAP:<one short question>

Rules:
- Do not use the target word as an answer.
- Prefer an existing concept when it fits.
- Use GAP only when the slot cannot be completed from the current context.
- The GAP question must be specific and reusable.
- No explanation outside the four lines.
""".strip(),
    )


def answer_prompt(
    tokenizer,
    word: str,
    question: str,
    context: list[str],
) -> str:
    context_text = (
        ", ".join(context)
        if context
        else "(none)"
    )

    return apply_chat(
        tokenizer,
        (
            "You answer a semantic learning question for a compact external "
            "memory. Reuse known concepts where possible. "
            "Return only comma-separated concepts."
        ),
        f"""
TARGET WORD:
{word}

QUESTION:
{question}

KNOWN CONCEPTS:
{context_text}

Return up to {MAX_CONCEPTS_PER_ANSWER} concepts.

Rules:
- Existing concepts may be returned exactly.
- A genuinely missing concept may be returned as NEW:<short concept>.
- No explanation.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def clean_concept(
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

    if len(value.split()) > 5:
        return None

    banned = (
        "the answer",
        "the word",
        "because",
        "explanation",
        "memory system",
    )

    if any(
        marker in value
        for marker in banned
    ):
        return None

    return value


def parse_task(
    text: str,
    context: set[str],
) -> dict[str, str]:
    result = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        match = re.match(
            r"^(CATEGORY|PROPERTY|USE|RELATION)\s*=\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        slot = match.group(1).upper()
        value = match.group(2).strip()

        if value.upper().startswith("GAP:"):
            question = value[4:].strip()

            if question:
                result[slot] = (
                    "GAP:",
                    question,
                )
            continue

        concept = clean_concept(
            value
        )

        if (
            concept is not None
            and concept in context
        ):
            result[slot] = concept

    return result


def parse_answer(
    text: str,
    context: set[str],
    target: str,
) -> list[str]:
    result = []

    for part in re.split(
        r",|;|\n|\|",
        text,
    ):
        raw = part.strip().lower()

        if not raw:
            continue

        learner_new = raw.startswith(
            "new:"
        )

        concept = clean_concept(
            raw
        )

        if concept is None:
            continue

        if concept == target:
            continue

        if learner_new:
            value = (
                "NEW:"
                + concept
            )
        else:
            if concept not in context:
                continue

            value = concept

        if value not in result:
            result.append(value)

        if len(result) >= MAX_CONCEPTS_PER_ANSWER:
            break

    return result


# ---------------------------------------------------------------------------
# Generation
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

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

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
# Evaluation
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

    precision = hits / len(predicted)
    recall = hits / len(gold)

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


def semantic_f1(
    memory: LearnedMemory,
    gold: HumanGold,
    words: list[str],
) -> float:
    values = []

    for word in words:
        gold_set = gold.gold(word)

        if not gold_set:
            continue

        predicted = set(
            memory.concepts_for_word(word)
        )

        values.append(
            f1(
                predicted,
                gold_set,
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
        "=== V115 TASK-DRIVEN SEMANTIC LEARNING ==="
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

    tokenizer, model, device = load_model()

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
    total_answer_calls = 0
    total_created = 0
    task_successes = 0
    task_failures = 0

    trace: list[dict] = []

    # -----------------------------------------------------------------------
    # Main task-learning loop.
    # -----------------------------------------------------------------------

    for round_index in range(
        1,
        ROUNDS + 1,
    ):
        print(
            f"\n=== TASK ROUND {round_index}/{ROUNDS} ===",
            flush=True,
        )

        task_jobs = []

        for start in range(
            0,
            len(words),
            BATCH_SIZE,
        ):
            batch_words = words[
                start:start + BATCH_SIZE
            ]

            batch_prompts = []
            batch_contexts = []

            for word in batch_words:
                context = memory.retrieve_context(
                    word,
                    lexical_index,
                )

                batch_contexts.append(
                    context
                )

                batch_prompts.append(
                    task_prompt(
                        tokenizer,
                        word,
                        context,
                    )
                )

            raw_tasks = generate_batch(
                tokenizer,
                model,
                device,
                batch_prompts,
            )

            for word, context, raw in zip(
                batch_words,
                batch_contexts,
                raw_tasks,
            ):
                parsed = parse_task(
                    raw,
                    set(context),
                )

                gaps = [
                    value[1]
                    for value in parsed.values()
                    if isinstance(value, tuple)
                    and value
                    and value[0] == "GAP:"
                ]

                known = [
                    value
                    for value in parsed.values()
                    if isinstance(value, str)
                ]

                if gaps:
                    task_failures += 1
                else:
                    task_successes += 1

                total_gaps += len(gaps)
                total_questions += len(gaps)

                if word in {
                    "hello",
                    "greeting",
                    "ability",
                    "abandon",
                    "water",
                    "music",
                }:
                    trace.append(
                        {
                            "round": round_index,
                            "word": word,
                            "context": context,
                            "raw_task": raw,
                            "parsed": {
                                key: (
                                    list(value)
                                    if isinstance(value, tuple)
                                    else value
                                )
                                for key, value
                                in parsed.items()
                            },
                            "gaps": gaps,
                            "known": known,
                        }
                    )

                for question in gaps:
                    answer_context = list(
                        dict.fromkeys(
                            context
                            + memory.top_used(16)
                        )
                    )[:CANDIDATE_LIMIT]

                    task_jobs.append(
                        (
                            word,
                            question,
                            set(answer_context),
                        )
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
                    f"TASK round={round_index} "
                    f"{processed:4d}/{len(words):4d} "
                    f"questions={total_questions} "
                    f"failures={task_failures} "
                    f"concepts={len(memory.concept_id_by_name)}",
                    flush=True,
                )

        # ---------------------------------------------------------------
        # Answer all mechanically generated gaps.
        # ---------------------------------------------------------------

        for start in range(
            0,
            len(task_jobs),
            BATCH_SIZE,
        ):
            batch_jobs = task_jobs[
                start:start + BATCH_SIZE
            ]

            answer_prompts = [
                answer_prompt(
                    tokenizer,
                    word,
                    question,
                    list(context),
                )
                for word, question, context
                in batch_jobs
            ]

            raw_answers = generate_batch(
                tokenizer,
                model,
                device,
                answer_prompts,
            )

            for (
                word,
                question,
                context,
            ), raw_answer in zip(
                batch_jobs,
                raw_answers,
            ):
                allowed = set(context)

                concepts = parse_answer(
                    raw_answer,
                    allowed,
                    word,
                )

                created = memory.add_concepts(
                    word,
                    concepts,
                )

                total_answer_calls += 1
                total_created += created

                if word in {
                    "hello",
                    "greeting",
                    "ability",
                    "abandon",
                    "water",
                    "music",
                }:
                    trace.append(
                        {
                            "round": round_index,
                            "word": word,
                            "question": question,
                            "answer_raw": raw_answer,
                            "answer_concepts": concepts,
                            "created": created,
                            "after": memory.concepts_for_word(
                                word
                            ),
                        }
                    )

            processed = min(
                start + BATCH_SIZE,
                len(task_jobs),
            )

            if (
                processed <= BATCH_SIZE
                or processed % PRINT_EVERY == 0
                or processed == len(task_jobs)
            ):
                print(
                    f"ANSWER round={round_index} "
                    f"{processed:4d}/{len(task_jobs):4d} "
                    f"new_concepts={total_created} "
                    f"memory={len(memory.concept_id_by_name)}",
                    flush=True,
                )

        memory.rebuild_co_usage()

        round_f1 = semantic_f1(
            memory,
            gold,
            words,
        )

        print(
            f"ROUND {round_index} COMPLETE "
            f"questions={total_questions} "
            f"answers={total_answer_calls} "
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
        "=== V115 SUMMARY ==="
    )

    print(
        "initial_concepts:",
        initial_concepts,
    )

    print(
        "final_concepts:",
        len(memory.concept_id_by_name),
    )

    print(
        "concept_growth:",
        len(memory.concept_id_by_name)
        - initial_concepts,
    )

    print(
        "task_successes:",
        task_successes,
    )

    print(
        "task_failures:",
        task_failures,
    )

    print(
        "questions_generated:",
        total_questions,
    )

    print(
        "answer_calls:",
        total_answer_calls,
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
        final_f1 - initial_f1,
    )

    print()

    print(
        "=== TRACE ==="
    )

    for item in trace[:50]:
        print(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )

    payload = {
        "experiment": "V115 task-driven semantic learning",
        "initial_concepts": initial_concepts,
        "final_concepts": len(
            memory.concept_id_by_name
        ),
        "concept_growth": (
            len(memory.concept_id_by_name)
            - initial_concepts
        ),
        "task_successes": task_successes,
        "task_failures": task_failures,
        "questions_generated": total_questions,
        "answer_calls": total_answer_calls,
        "new_concepts": total_created,
        "semantic_f1_before": initial_f1,
        "semantic_f1_after": final_f1,
        "semantic_f1_delta": (
            final_f1 - initial_f1
        ),
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

    report_path = (
        ROOT
        / "results"
        / "v115_task_driven_semantic_learning.json"
    )

    report_path.write_text(
        json.dumps(
            payload,
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
        report_path,
    )

    print(
        "=== V115 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
