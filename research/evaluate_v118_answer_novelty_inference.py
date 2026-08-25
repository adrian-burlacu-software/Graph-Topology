from __future__ import annotations

"""
V117 — DETERMINISTIC SLOT INQUIRY

This is the direct fix to V116.

The LLM does NOT generate the question.

For every target word, we force a constrained semantic reconstruction over
four explicit slots:

    CATEGORY
    PROPERTY
    USE
    RELATION

The LLM may ONLY answer a slot with an exact concept from the current graph
context. Anything else is discarded by the program.

Therefore:

    valid answer in slot   -> slot satisfied
    invalid / missing      -> mechanical semantic GAP

The program then asks a deterministic question for the missing slot:

    CATEGORY  -> "What kind of thing is X?"
    PROPERTY  -> "What property describes X?"
    USE       -> "What is X used for?"
    RELATION  -> "What is X related to?"

The frozen LLM answers ONLY that question.

The graph stores:
    existing concept
    or NEW:<concept>

No LLM-generated question is used.

The semantic corpus is evaluation-only.

This is the experiment that isolates:
    GAP DETECTION  = program
    QUESTION       = program
    ANSWER         = frozen LLM
    MEMORY         = Graph-Topology

Starts from V111 memory.
Does not modify V111.
"""

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "llm" / "SmolLM2-360M-Instruct"

INPUT_MEMORY = ROOT / "results" / "v111_compact_semantic_memory.json"

OUTPUT_MEMORY = ROOT / "results" / "v117_deterministic_slot_memory.json"

OUTPUT_REPORT = ROOT / "results" / "v117_deterministic_slot_inquiry.json"

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"

SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

BATCH_SIZE = 128

ROUNDS = 2

MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 20

CANDIDATE_LIMIT = 32

# How many slots need valid existing concepts before we regard the current
# representation as satisfactory.
MIN_SATISFIED_SLOTS = 4

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
        raise RuntimeError("No dictionary words found.")

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
# Human gold — evaluation only
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
# Learned memory
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
            for word, ids in payload.get(
                "word_concepts",
                {},
            ).items()
        }

        self.words_by_concept: dict[int, set[str]] = defaultdict(set)

        for word, identifiers in self.word_concepts.items():
            for identifier in identifiers:
                self.words_by_concept[identifier].add(word)

        self.learner_generated = set(
            payload.get("learner_generated", [])
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

    def concepts_for_word(self, word: str) -> list[str]:
        return [
            self.concept_name_by_id[identifier]
            for identifier in self.word_concepts.get(word, [])
            if identifier in self.concept_name_by_id
        ]

    def top_used(self, limit: int) -> list[str]:
        ranked = sorted(
            self.usage.items(),
            key=lambda item: (
                -item[1],
                self.concept_name_by_id[item[0]],
            ),
        )

        return [
            self.concept_name_by_id[identifier]
            for identifier, _count in ranked[:limit]
        ]

    def retrieve_context(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
        scores = Counter()

        for neighbor in lexical_index.get(word, []):
            for identifier in self.word_concepts.get(neighbor, []):
                scores[identifier] += 1.0

        for identifier, _score in scores.most_common(16):
            for related, count in self.co_usage.get(
                identifier,
                Counter(),
            ).most_common(8):
                scores[related] += 0.25 * count

        target_ids = set(
            self.word_concepts.get(word, [])
        )

        for identifier in target_ids:
            scores.pop(identifier, None)

        for concept in self.top_used(16):
            identifier = self.concept_id_by_name.get(concept)

            if (
                identifier is not None
                and identifier not in target_ids
            ):
                scores[identifier] += 0.5

        return [
            self.concept_name_by_id[identifier]
            for identifier, _score in scores.most_common(
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

            identifier = self.concept_id_by_name.get(concept)

            if identifier is None:
                identifier = len(self.concept_id_by_name)

                self.concept_id_by_name[concept] = identifier
                self.concept_name_by_id[identifier] = concept

                created += 1

                if learner_new:
                    self.learner_generated.add(concept)

            if identifier not in ids:
                ids.append(identifier)

            self.usage[identifier] += 1
            self.words_by_concept[identifier].add(word)

        self.word_concepts[word] = ids

        return created

    def save(self, path: Path) -> None:
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

    print("device:", device, flush=True)

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

SLOT_SYSTEM = (
    "You are a strict semantic classifier. "
    "For each slot choose EXACTLY ONE concept from the supplied vocabulary. "
    "If no supplied concept fits, write UNKNOWN. "
    "Never invent a concept. "
    "Return exactly four labeled lines."
)


def slot_prompt(
    tokenizer,
    word: str,
    context: list[str],
) -> str:
    return apply_chat(
        tokenizer,
        SLOT_SYSTEM,
        f"""
TARGET:
{word}

VOCABULARY:
{", ".join(context) if context else "(none)"}

Return EXACTLY:

CATEGORY=<one vocabulary item or UNKNOWN>
PROPERTY=<one vocabulary item or UNKNOWN>
USE=<one vocabulary item or UNKNOWN>
RELATION=<one vocabulary item or UNKNOWN>

Rules:
- The value after = must be EXACTLY one item from VOCABULARY or UNKNOWN.
- Never use the target word.
- No explanation.
""".strip(),
    )


QUESTION_SYSTEM = (
    "You generate deterministic semantic questions. "
    "Answer only with one short question."
)


QUESTION_TEMPLATES = {
    "CATEGORY": "What kind of thing is {word}?",
    "PROPERTY": "What property best describes {word}?",
    "USE": "What is {word} used for?",
    "RELATION": "What is {word} related to?",
}


def deterministic_question(
    slot: str,
    word: str,
) -> str:
    return QUESTION_TEMPLATES[slot].format(
        word=word
    )


def answer_prompt(
    tokenizer,
    word: str,
    slot: str,
    question: str,
    context: list[str],
) -> str:
    return apply_chat(
        tokenizer,
        (
            "You answer one semantic question for an external memory. "
            "Reuse known concepts when possible. "
            "If the answer requires a missing concept, output NEW:<short concept>. "
            "Return exactly one concept."
        ),
        f"""
TARGET:
{word}

SLOT:
{slot}

QUESTION:
{question}

KNOWN CONCEPTS:
{", ".join(context) if context else "(none)"}

Return exactly ONE concept.

Rules:
- Existing concept must match KNOWN CONCEPTS exactly.
- Otherwise use NEW:<short concept>.
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

    value = value.strip(
        " \t\r\n.,;:!?\"'`()[]{}"
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

    # Keep short concepts, not whole generated explanations.
    if len(value.split()) > 4:
        return None

    banned = (
        "the answer",
        "the word",
        "explanation",
        "because",
        "description",
        "assistant",
        "return exactly",
        "no explanation",
        "known concepts",
        "vocabulary",
    )

    if any(
        marker in value
        for marker in banned
    ):
        return None

    return value


def parse_slots(
    text: str,
    allowed: set[str],
    target: str,
) -> dict[str, str | None]:
    result = {
        "CATEGORY": None,
        "PROPERTY": None,
        "USE": None,
        "RELATION": None,
    }

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
        raw_value = match.group(2).strip()

        if raw_value.upper() == "UNKNOWN":
            result[slot] = None
            continue

        value = normalize(
            raw_value
        )

        if (
            value is not None
            and value != target
            and value in allowed
        ):
            result[slot] = value
        else:
            result[slot] = None

    return result


def parse_answer(
    text: str,
    allowed: set[str],
    target: str,
) -> str | None:
    """
    Mechanically interpret the model's one-concept answer.

    The model does NOT have to emit NEW:.

        known concept          -> reuse
        short unknown concept  -> NEW:<concept>
        obvious meta/junk      -> reject

    This is the key V118 change.
    """
    if not text:
        return None

    # Prefer the first non-empty line and first comma/semicolon-separated item.
    line = next(
        (
            item.strip()
            for item in text.splitlines()
            if item.strip()
        ),
        "",
    )

    if not line:
        return None

    line = re.split(
        r"[,;|]",
        line,
        maxsplit=1,
    )[0].strip()

    # Strip common labels the tiny instruction model may echo.
    line = re.sub(
        r"^(answer|concept|response|result)\s*:\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )

    explicit_new = line.lower().startswith(
        "new:"
    )

    raw_value = (
        line[4:].strip()
        if explicit_new
        else line
    )

    value = normalize(
        raw_value
    )

    if value is None:
        return None

    if value == target:
        return None

    # Exact reuse.
    if value in allowed:
        return value

    # The model doesn't need to understand the NEW protocol. The program
    # infers novelty from the vocabulary boundary.
    if explicit_new or value not in allowed:
        return (
            "NEW:"
            + value
        )

    return None


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
        gold_set = gold.gold(word)

        if not gold_set:
            continue

        values.append(
            f1(
                set(
                    memory.concepts_for_word(
                        word
                    )
                ),
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
        "=== V117 DETERMINISTIC SLOT INQUIRY ==="
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
        INPUT_MEMORY
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
    )

    print(
        "initial_semantic_f1:",
        initial_f1,
    )

    total_gaps = 0
    total_questions = 0
    total_answer_calls = 0
    total_created = 0
    total_reused_answers = 0
    total_rejected_answers = 0

    slot_gap_counts = Counter()
    successful_slot_answers = Counter()

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
        # Phase 1 — strict slot classification.
        # ---------------------------------------------------------------

        answer_jobs = []

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
                slot_prompt(
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
                parsed = parse_slots(
                    raw,
                    set(context),
                    word,
                )

                missing = [
                    slot
                    for slot, value
                    in parsed.items()
                    if value is None
                ]

                for slot in missing:
                    slot_gap_counts[
                        slot
                    ] += 1

                    question = deterministic_question(
                        slot,
                        word,
                    )

                    answer_jobs.append(
                        (
                            word,
                            slot,
                            question,
                            context,
                        )
                    )

                for slot, value in parsed.items():
                    if value is not None:
                        successful_slot_answers[
                            slot
                        ] += 1

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "phase": "SLOTS",
                            "round": round_index,
                            "word": word,
                            "context": context,
                            "raw": raw,
                            "parsed": parsed,
                            "missing": missing,
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
                total_current_gaps = sum(
                    slot_gap_counts.values()
                )

                print(
                    f"SLOTS "
                    f"round={round_index} "
                    f"{processed:4d}/{len(words):4d} "
                    f"gaps={total_current_gaps} "
                    f"memory={len(memory.concept_id_by_name)}",
                    flush=True,
                )

        total_gaps += len(answer_jobs)

        # ---------------------------------------------------------------
        # Phase 2 — deterministic question + single concept answer.
        # ---------------------------------------------------------------

        for start in range(
            0,
            len(answer_jobs),
            BATCH_SIZE,
        ):
            batch_jobs = answer_jobs[
                start:start + BATCH_SIZE
            ]

            prompts = [
                answer_prompt(
                    tokenizer,
                    word,
                    slot,
                    question,
                    context,
                )
                for word, slot, question, context
                in batch_jobs
            ]

            raw_answers = generate_batch(
                tokenizer,
                model,
                device,
                prompts,
            )

            for (
                (word, slot, question, context),
                raw_answer,
            ) in zip(
                batch_jobs,
                raw_answers,
            ):
                total_answer_calls += 1

                parsed = parse_answer(
                    raw_answer,
                    set(context),
                    word,
                )

                if parsed is None:
                    total_rejected_answers += 1
                    continue

                if parsed.startswith("NEW:"):
                    total_created += 1
                else:
                    total_reused_answers += 1

                created = memory.add_concepts(
                    word,
                    [parsed],
                )

                total_questions += 1

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "phase": "ANSWER",
                            "round": round_index,
                            "word": word,
                            "slot": slot,
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
                len(answer_jobs),
            )

            if (
                processed <= BATCH_SIZE
                or processed % PRINT_EVERY == 0
                or processed == len(answer_jobs)
            ):
                print(
                    f"ANSWERS "
                    f"round={round_index} "
                    f"{processed:4d}/{len(answer_jobs):4d} "
                    f"created={total_created} "
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
            f"gaps={len(answer_jobs)} "
            f"answered={total_questions} "
            f"new={total_created} "
            f"concepts={len(memory.concept_id_by_name)} "
            f"semantic_f1={round_f1:.4f}",
            flush=True,
        )

        memory.save(
            OUTPUT_MEMORY
        )

    final_f1 = semantic_f1(
        memory,
        gold,
        words,
    )

    print()
    print(
        "=== V117 SUMMARY ==="
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
        "total_gaps:",
        total_gaps,
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
        "reused_answers:",
        total_reused_answers,
    )

    print(
        "rejected_answers:",
        total_rejected_answers,
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
        "slot_gaps:",
        dict(slot_gap_counts),
    )

    print(
        "slot_successes:",
        dict(successful_slot_answers),
    )

    print()
    print(
        "=== TRACE ==="
    )

    for item in trace[:120]:
        print(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )

    report = {
        "experiment": (
            "V117 deterministic slot inquiry"
        ),
        "initial_concepts": initial_concepts,
        "final_concepts": len(
            memory.concept_id_by_name
        ),
        "concept_growth": (
            len(memory.concept_id_by_name)
            - initial_concepts
        ),
        "total_gaps": total_gaps,
        "questions_generated": total_questions,
        "answer_calls": total_answer_calls,
        "new_concepts": total_created,
        "reused_answers": total_reused_answers,
        "rejected_answers": total_rejected_answers,
        "semantic_f1_before": initial_f1,
        "semantic_f1_after": final_f1,
        "semantic_f1_delta": (
            final_f1 - initial_f1
        ),
        "slot_gaps": dict(slot_gap_counts),
        "slot_successes": dict(successful_slot_answers),
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

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    memory.save(
        OUTPUT_MEMORY
    )

    print()
    print(
        "saved_memory:",
        OUTPUT_MEMORY,
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
        "=== V117 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
