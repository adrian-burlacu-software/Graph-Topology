from __future__ import annotations

"""
V114 — AUTONOMOUS SEMANTIC INQUIRY

This experiment starts from the V111 learned semantic memory and tests a
different capability:

    Can a frozen LLM notice that its external memory is incomplete,
    formulate a useful question, answer that question, and consolidate
    the answer back into the graph?

No semantic corpus is used to construct candidates during the inquiry loop.
semantics-large.csv is used ONLY AFTERWARD for evaluation.

Pipeline
--------
V111 memory
    ↓
word presented with its OWN memory edges hidden
    ↓
graph retrieves related concepts from other words
    ↓
frozen LLM decides:
    RECALL:<concepts>
    OR
    ASK:<question>
    ↓
if ASK:
    frozen LLM answers the question using current graph vocabulary
    ↓
NEW:<concept> allowed when needed
    ↓
graph expands
    ↓
repeat up to INQUIRY_ROUNDS

The intended emergent behavior is something like:

    hello
      ↓
    incomplete memory
      ↓
    QUESTION: what is a greeting?
      ↓
    ANSWER: greeting, communication, social, ...
      ↓
    graph stores greeting
      ↓
    future words can reuse greeting

This is NOT claiming the model has subjective curiosity. The experiment tests
an operational form of self-directed semantic inquiry.
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

OUTPUT_PATH = ROOT / "results" / "v114_autonomous_semantic_inquiry.json"

BATCH_SIZE = 128
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 32

INQUIRY_ROUNDS = 3
CANDIDATE_LIMIT = 32
MAX_CONCEPTS = 8

# Evaluate the whole dictionary.
MAX_WORDS = None

# For speed, only print examples for a small sample.
EXAMPLE_WORDS = {
    "hello",
    "greeting",
    "abandon",
    "ability",
    "animal",
    "water",
    "music",
    "chair",
    "car",
}


# ---------------------------------------------------------------------------
# Dictionary / semantic gold
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

    if MAX_WORDS is not None:
        result = result[:MAX_WORDS]

    if not result:
        raise RuntimeError("No dictionary words found.")

    return result



# ---------------------------------------------------------------------------
# Fast precomputed lexical index
# ---------------------------------------------------------------------------

def build_lexical_index(
    words: list[str],
) -> dict[str, list[str]]:
    """
    Build lexical neighbors once.

    This prevents an O(N^2) Python scan inside every inquiry.
    """
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

                if weight <= 0:
                    try:
                        frequency = float(
                            row.get(
                                "frequency_translated",
                                0.0,
                            )
                        )
                        n = float(row.get("n", 0.0))
                        if n > 0:
                            weight = frequency / n
                    except (TypeError, ValueError):
                        weight = 0.0

                if weight > 0:
                    self.cue_features[cue][feature] += weight

    def gold(self, word: str, limit: int = MAX_CONCEPTS) -> set[str]:
        return set(
            feature
            for feature, _weight
            in self.cue_features.get(word, Counter()).most_common(limit)
        )


# ---------------------------------------------------------------------------
# Learned V111 graph
# ---------------------------------------------------------------------------

class LearnedMemory:
    def __init__(self, path: Path) -> None:
        payload = json.loads(
            path.read_text(encoding="utf-8")
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
            in payload.get("word_concepts", {}).items()
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

    def add_concepts(
        self,
        word: str,
        concepts: list[str],
    ) -> int:
        created = 0

        ids = list(
            self.word_concepts.get(word, [])
        )

        for raw in concepts:
            learner_new = raw.startswith("NEW:")
            concept = (
                raw[4:].strip().lower()
                if learner_new
                else raw.strip().lower()
            )

            concept = re.sub(r"\s+", " ", concept)

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

    def lexical_neighbors(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
        limit: int = 12,
    ) -> list[str]:
        return lexical_index.get(
            word,
            [],
        )[:limit]

    def retrieve_without_own_edges(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
        """
        Retrieve context using only graph structure while masking the target
        word's own stored concept edges.
        """
        scores = Counter()

        neighbors = lexical_index.get(
            word,
            [],
        )

        # Concept evidence from lexical neighbors.
        for neighbor in neighbors:
            for identifier in self.word_concepts.get(
                neighbor,
                [],
            ):
                scores[identifier] += 1.0

        # Concept co-usage provides a second graph-only signal.
        for identifier, _score in scores.most_common(16):
            for related, count in self.co_usage.get(
                identifier,
                Counter(),
            ).most_common(8):
                scores[related] += 0.25 * count

        # Mask the target's own learned edges.
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

        # Small global graph fallback.
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

    def save(self, path: Path) -> None:
        payload = {
            "concept_id_by_name": self.concept_id_by_name,
            "usage": {
                str(identifier): count
                for identifier, count in self.usage.items()
            },
            "words_by_concept": {
                str(identifier): sorted(words)
                for identifier, words in self.words_by_concept.items()
            },
            "word_concepts": self.word_concepts,
            "learner_generated": sorted(self.learner_generated),
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
# LLM
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


def chat_prompt(
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


INQUIRY_SYSTEM = (
    "You are an external-memory learner. "
    "The memory may be incomplete. "
    "If it is insufficient, formulate one useful semantic question. "
    "Return only the requested machine-readable format."
)


def make_inquiry_prompt(
    tokenizer,
    word: str,
    candidates: list[str],
) -> str:
    return chat_prompt(
        tokenizer,
        INQUIRY_SYSTEM,
        f"""
TARGET WORD:
{word}

CURRENT MEMORY CONTEXT:
{", ".join(candidates) if candidates else "(none)"}

Decide whether the current memory is sufficient.

Return exactly ONE line in one of these forms:

RECALL: concept, concept
ASK: one short question that would help define the target

Rules:
- Ask only if important meaning is missing.
- Prefer a question about a reusable concept or category.
- Do not mention the memory system.
- Do not explain your decision.
""".strip(),
    )


def make_answer_prompt(
    tokenizer,
    word: str,
    question: str,
    candidates: list[str],
) -> str:
    return chat_prompt(
        tokenizer,
        (
            "You answer semantic questions for a compact external memory. "
            "Use existing concepts whenever they fit. "
            "Only introduce NEW:concept when necessary. "
            "Return only a comma-separated concept list."
        ),
        f"""
TARGET:
{word}

QUESTION:
{question}

KNOWN CONCEPTS:
{", ".join(candidates) if candidates else "(none)"}

Return up to {MAX_CONCEPTS} items.

Rules:
- Reuse known concepts whenever possible.
- A novel concept must be written as NEW:<short concept>.
- No explanation.
- No sentence.
""".strip(),
    )


def clean_concept(value: str) -> str | None:
    value = value.strip().lower()

    value = re.sub(
        r"^new\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"[^a-z0-9 ?!\-]",
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
        item in value
        for item in banned
    ):
        return None

    return value


def parse_inquiry(text: str) -> tuple[str, str | list[str]]:
    line = text.strip()

    line = line.splitlines()[0] if line else ""

    if line.upper().startswith("ASK:"):
        return (
            "ask",
            line[4:].strip(),
        )

    if line.upper().startswith("RECALL:"):
        raw = line[7:]
        concepts = []

        for part in re.split(
            r",|;|\|",
            raw,
        ):
            concept = clean_concept(part)

            if concept:
                concepts.append(concept)

            if len(concepts) >= MAX_CONCEPTS:
                break

        return (
            "recall",
            concepts,
        )

    # Salvage malformed output as a question if it contains a question mark.
    if "?" in line:
        return (
            "ask",
            line,
        )

    return (
        "recall",
        [],
    )


def parse_answer(
    text: str,
    candidates: set[str],
    target: str,
) -> list[str]:
    concepts = []

    for part in re.split(
        r",|;|\n|\|",
        text,
    ):
        raw = part.strip().lower()

        if not raw:
            continue

        learner_new = raw.startswith("new:")

        concept = clean_concept(raw)

        if concept is None:
            continue

        if concept == target:
            continue

        if learner_new:
            value = "NEW:" + concept
        else:
            # Only exact known concepts may be reused.
            if concept not in candidates:
                continue
            value = concept

        if value not in concepts:
            concepts.append(value)

        if len(concepts) >= MAX_CONCEPTS:
            break

    return concepts


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

    for row in range(output.shape[0]):
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

    hits = len(predicted & gold)

    precision = hits / len(predicted)
    recall = hits / len(gold)

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def evaluate_memory(
    memory: LearnedMemory,
    words: list[str],
    gold: HumanGold,
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
        "=== V114 AUTONOMOUS SEMANTIC INQUIRY ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    dictionary_set = set(words)

    lexical_index = build_lexical_index(
        words
    )

    print(
        "lexical_index_ready:",
        len(lexical_index),
        flush=True,
    )

    graph = LearnedMemory(
        MEMORY_PATH
    )

    gold = HumanGold()
    gold.load(
        SEMANTICS_PATH
    )

    tokenizer, model, device = load_model()

    before_f1 = evaluate_memory(
        graph,
        words,
        gold,
    )

    print(
        "initial_graph_concepts:",
        len(graph.concept_id_by_name),
    )

    print(
        "initial_semantic_f1:",
        before_f1,
    )

    print(
        "dictionary_words:",
        len(words),
    )

    print()

    # Metrics.
    questions_generated = 0
    recall_decisions = 0
    inquiry_round_counts = Counter()
    novel_concepts = 0
    words_changed = 0
    answer_calls = 0

    example_log = []

    # We need per-round temporary states.
    states = {
        word: {
            "active": True,
            "rounds": 0,
            "questions": [],
        }
        for word in words
    }

    for round_index in range(
        INQUIRY_ROUNDS
    ):
        active_words = [
            word
            for word in words
            if states[word]["active"]
        ]

        if not active_words:
            break

        print()
        print(
            f"=== INQUIRY ROUND {round_index + 1}/{INQUIRY_ROUNDS} "
            f"active={len(active_words)} ===",
            flush=True,
        )

        # -----------------------------------------------------------
        # Phase 1: decide recall vs ask.
        # -----------------------------------------------------------
        inquiry_prompts = []
        inquiry_words = []
        candidate_sets = {}

        for word in active_words:
            candidates = graph.retrieve_without_own_edges(
                word,
                lexical_index,
            )

            candidate_sets[word] = set(
                candidates
            )

            inquiry_prompts.append(
                make_inquiry_prompt(
                    tokenizer,
                    word,
                    candidates,
                )
            )

            inquiry_words.append(
                word
            )

        decisions = []

        for start in range(
            0,
            len(inquiry_prompts),
            BATCH_SIZE,
        ):
            batch_prompts = inquiry_prompts[
                start:start + BATCH_SIZE
            ]

            batch_words = inquiry_words[
                start:start + BATCH_SIZE
            ]

            raw = generate_batch(
                tokenizer,
                model,
                device,
                batch_prompts,
            )

            for word, text in zip(
                batch_words,
                raw,
            ):
                decisions.append(
                    (
                        word,
                        parse_inquiry(
                            text
                        ),
                    )
                )

        # -----------------------------------------------------------
        # Phase 2: answers to self-generated questions.
        # -----------------------------------------------------------
        answer_jobs = []

        for word, (kind, payload) in decisions:
            states[word]["rounds"] += 1

            if kind == "ask":
                questions_generated += 1

                question = str(payload)

                states[word]["questions"].append(
                    question
                )

                answer_context = graph.top_used(
                    CANDIDATE_LIMIT
                )

                answer_jobs.append(
                    (
                        word,
                        question,
                        candidate_sets[word],
                        answer_context,
                    )
                )

            else:
                recall_decisions += 1
                states[word]["active"] = False

                if word in EXAMPLE_WORDS:
                    example_log.append(
                        {
                            "word": word,
                            "round": round_index + 1,
                            "decision": "RECALL",
                            "known": graph.concepts_for_word(word),
                            "question": None,
                        }
                    )

        # -----------------------------------------------------------
        # Phase 3: answer the questions.
        # -----------------------------------------------------------
        if answer_jobs:
            answer_prompts = [
                make_answer_prompt(
                    tokenizer,
                    word,
                    question,
                    context,
                )
                for word, question, _candidate_set, context
                in answer_jobs
            ]

            for start in range(
                0,
                len(answer_prompts),
                BATCH_SIZE,
            ):
                batch_prompts = answer_prompts[
                    start:start + BATCH_SIZE
                ]

                batch_jobs = answer_jobs[
                    start:start + BATCH_SIZE
                ]

                raw_answers = generate_batch(
                    tokenizer,
                    model,
                    device,
                    batch_prompts,
                )

                for (
                    (word, question, candidate_set, context),
                    raw_answer,
                ) in zip(
                    batch_jobs,
                    raw_answers,
                ):
                    answer_calls += 1

                    # Allow all context concepts plus concepts already stored in
                    # the graph. NEW: is separately allowed.
                    allowed = set(
                        context
                        + graph.concepts_for_word(word)
                    )

                    parsed = parse_answer(
                        raw_answer,
                        allowed,
                        word,
                    )

                    created = graph.add_concepts(
                        word,
                        parsed,
                    )

                    novel_concepts += created

                    if created:
                        words_changed += 1

                    # A successful answer finishes this round. The next round
                    # can still ask another question if active.
                    states[word]["active"] = True

                    if word in EXAMPLE_WORDS:
                        example_log.append(
                            {
                                "word": word,
                                "round": round_index + 1,
                                "decision": "ASK",
                                "known_before": sorted(candidate_set),
                                "question": question,
                                "answer": parsed,
                                "created": created,
                                "known_after": graph.concepts_for_word(
                                    word
                                ),
                            }
                        )

        # Stop words that had no question/answer path.
        for word, (kind, _payload) in decisions:
            if kind == "recall":
                continue

            # If a question was generated but the answer created no new
            # information, keep one more chance; otherwise eventually stop.
            if states[word]["rounds"] >= INQUIRY_ROUNDS:
                states[word]["active"] = False

        graph.rebuild_co_usage()

        round_f1 = evaluate_memory(
            graph,
            words,
            gold,
        )

        print(
            f"round={round_index + 1} "
            f"questions={questions_generated} "
            f"recalls={recall_decisions} "
            f"answer_calls={answer_calls} "
            f"new_concepts={novel_concepts} "
            f"graph_concepts={len(graph.concept_id_by_name)} "
            f"semantic_f1={round_f1:.4f}",
            flush=True,
        )

    after_f1 = evaluate_memory(
        graph,
        words,
        gold,
    )

    inquiry_rate = (
        questions_generated
        / max(
            1,
            len(words)
            * INQUIRY_ROUNDS,
        )
    )

    print()
    print(
        "=== V114 SUMMARY ==="
    )

    print(
        "initial_graph_concepts:",
        len(
            LearnedMemory(
                MEMORY_PATH
            ).concept_id_by_name
        ),
    )

    print(
        "final_graph_concepts:",
        len(
            graph.concept_id_by_name
        ),
    )

    print(
        "novel_concepts_created:",
        novel_concepts,
    )

    print(
        "questions_generated:",
        questions_generated,
    )

    print(
        "answer_calls:",
        answer_calls,
    )

    print(
        "recall_decisions:",
        recall_decisions,
    )

    print(
        "question_rate:",
        inquiry_rate,
    )

    print(
        "semantic_f1_before:",
        before_f1,
    )

    print(
        "semantic_f1_after:",
        after_f1,
    )

    print(
        "semantic_f1_delta:",
        after_f1 - before_f1,
    )

    print()

    print(
        "=== AUTONOMOUS QUESTION EXAMPLES ==="
    )

    shown = set()

    for item in example_log:
        if item["decision"] != "ASK":
            continue

        word = item["word"]

        if word in shown:
            continue

        shown.add(word)

        print(
            f"{word}:"
        )
        print(
            "  question:",
            item.get("question"),
        )
        print(
            "  answer:",
            item.get("answer"),
        )
        print(
            "  created:",
            item.get("created"),
        )
        print()

        if len(shown) >= 30:
            break

    # Save full state + selected example trace.
    payload = {
        "experiment": "V114 autonomous semantic inquiry",
        "initial_semantic_f1": before_f1,
        "final_semantic_f1": after_f1,
        "semantic_f1_delta": after_f1 - before_f1,
        "questions_generated": questions_generated,
        "answer_calls": answer_calls,
        "recall_decisions": recall_decisions,
        "question_rate": inquiry_rate,
        "novel_concepts_created": novel_concepts,
        "final_concept_count": len(
            graph.concept_id_by_name
        ),
        "example_log": example_log,
        "word_concepts": graph.word_concepts,
        "concept_id_by_name": graph.concept_id_by_name,
        "usage": {
            str(identifier): count
            for identifier, count in graph.usage.items()
        },
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

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V114 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
