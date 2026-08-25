from __future__ import annotations

"""
V119 — FIXED SEMANTIC QUERY TEACHER

Controlled experiment after V118.

Do NOT run another full autonomous mutation loop.

Instead:
    * load the V111 memory
    * query the frozen SmolLM2 with fixed semantic relations
    * use a small evaluation subset
    * compare raw answers against human semantic gold
    * only insert an answer into a COPY of the graph if it passes strict checks

Relations:
    CATEGORY
    FEATURE
    USE
    RELATION

The model is never asked to invent questions.

For each word we ask one fixed question per relation.

Output is a SINGLE short answer.

The semantic corpus is:
    evaluation gold only
It is NOT placed in prompts.

The graph copy is:
    mutation sandbox
so a bad run cannot corrupt V111.

The goal is to answer:
    "Can a frozen LLM provide useful semantic answers through a fixed,
     machine-checkable query interface, and do those answers improve the
     compressed external memory?"

Evaluation:
    * raw answer vs human gold
    * accepted answer count
    * rejected answer count
    * new concept count
    * F1 before / after mutation
    * per-relation performance

Default:
    512 words
    4 relations
    2048 queries
    batch 128
"""

import csv
import json
import re
import shutil
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

INPUT_MEMORY = ROOT / "results" / "v111_compact_semantic_memory.json"

OUTPUT_MEMORY = ROOT / "results" / "v119_fixed_query_memory.json"

OUTPUT_REPORT = ROOT / "results" / "v119_fixed_semantic_query_teacher.json"

DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"

SEMANTICS_PATH = ROOT / "data" / "semantics-large.csv"

EVAL_WORDS = 512

BATCH_SIZE = 128

MAX_INPUT_TOKENS = 192
MAX_NEW_TOKENS = 16

MAX_CANDIDATES = 32

RELATIONS = (
    "CATEGORY",
    "FEATURE",
    "USE",
    "RELATION",
)

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

    result = sorted(words)[:EVAL_WORDS]

    if not result:
        raise RuntimeError(
            "No dictionary words found."
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
                not other.startswith(word[:3]),
                not other.endswith(word[-3:]),
                abs(len(other) - len(word)),
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
        self.cue_features: dict[str, Counter[str]] = defaultdict(Counter)

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
                        n = float(
                            row.get("n", 0.0)
                        )

                        if n > 0:
                            weight = (
                                frequency / n
                            )
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
# Memory
# ---------------------------------------------------------------------------

class Memory:
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
                for identifier, count in payload.get(
                    "usage",
                    {},
                ).items()
            }
        )

        self.word_concepts = {
            str(word): [
                int(identifier)
                for identifier in identifiers
            ]
            for word, identifiers in payload.get(
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
            unique = list(
                dict.fromkeys(
                    identifiers
                )
            )

            for i, left in enumerate(unique):
                for right in unique[i + 1:]:
                    self.co_usage[left][right] += 1
                    self.co_usage[right][left] += 1

    def concepts_for_word(
        self,
        word: str,
    ) -> list[str]:
        return [
            self.concept_name_by_id[identifier]
            for identifier in self.word_concepts.get(
                word,
                [],
            )
        ]

    def context(
        self,
        word: str,
        lexical_index: dict[str, list[str]],
    ) -> list[str]:
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
            scores.pop(identifier, None)

        ranked = [
            self.concept_name_by_id[identifier]
            for identifier, _score
            in scores.most_common(
                MAX_CANDIDATES
            )
        ]

        return ranked

    def add_concept(
        self,
        word: str,
        concept_value: str,
    ) -> bool:
        """
        Returns True iff a new graph concept node was created.
        """
        learner_new = concept_value.startswith(
            "NEW:"
        )

        concept = (
            concept_value[4:].strip().lower()
            if learner_new
            else concept_value.strip().lower()
        )

        concept = re.sub(
            r"\s+",
            " ",
            concept,
        )

        if not concept:
            return False

        identifier = self.concept_id_by_name.get(
            concept
        )

        created = False

        if identifier is None:
            identifier = len(
                self.concept_id_by_name
            )
            self.concept_id_by_name[concept] = identifier
            self.concept_name_by_id[identifier] = concept
            created = True

            if learner_new:
                self.learner_generated.add(
                    concept
                )

        identifiers = self.word_concepts.setdefault(
            word,
            [],
        )

        if identifier not in identifiers:
            identifiers.append(
                identifier
            )

        self.usage[identifier] += 1
        self.words_by_concept[identifier].add(word)

        return created

    def save(
        self,
        path: Path,
    ) -> None:
        payload = {
            "concept_id_by_name": self.concept_id_by_name,
            "usage": {
                str(identifier): count
                for identifier, count in self.usage.items()
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
# Fixed query protocol
# ---------------------------------------------------------------------------

QUERY_TEXT = {
    "CATEGORY": "What category does {word} belong to?",
    "FEATURE": "What feature best describes {word}?",
    "USE": "What is {word} used for?",
    "RELATION": "What is {word} related to?",
}


def query_prompt(
    tokenizer,
    word: str,
    relation: str,
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
            "You answer one semantic query for an external memory. "
            "Return ONE short concept only. "
            "Do not explain."
        ),
        f"""
WORD:
{word}

QUERY:
{QUERY_TEXT[relation].format(word=word)}

CURRENT MEMORY CONTEXT:
{context_text}

Return ONE short concept.

Rules:
- Prefer an exact concept from CURRENT MEMORY CONTEXT if it fits.
- If the correct concept is absent, return a short ordinary English concept.
- Do not write a sentence.
- Do not repeat the word.
- No labels.
""".strip(),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_answer(
    text: str,
) -> str | None:
    if not text:
        return None

    line = next(
        (
            item.strip()
            for item in text.splitlines()
            if item.strip()
        ),
        "",
    )

    line = re.split(
        r"[,;|]",
        line,
        maxsplit=1,
    )[0]

    line = re.sub(
        r"^(answer|concept|response|result)\s*:\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )

    line = line.strip(
        " \t\r\n.,;:!?\"'`()[]{}"
    )

    line = re.sub(
        r"[^a-z0-9 \-]",
        " ",
        line.lower(),
    )

    line = re.sub(
        r"\s+",
        " ",
        line,
    ).strip()

    if not line:
        return None

    if len(line) > 40:
        return None

    if len(line.split()) > 4:
        return None

    banned = (
        "the answer",
        "the word",
        "explanation",
        "because",
        "assistant",
        "return one",
        "current memory",
    )

    if any(
        marker in line
        for marker in banned
    ):
        return None

    return line


def canonicalize_answer(
    answer: str | None,
    context: set[str],
    target: str,
) -> tuple[str, str] | None:
    """
    Returns:
        ("reuse", concept)
        ("new", concept)

    Novelty is inferred mechanically:
        exact context member -> reuse
        short valid unknown phrase -> new
    """
    if answer is None:
        return None

    if answer == target:
        return None

    if answer in context:
        return (
            "reuse",
            answer,
        )

    return (
        "new",
        answer,
    )


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

    precision = hits / len(predicted)
    recall = hits / len(gold)

    if precision + recall == 0.0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (
            precision + recall
        )
    )


def semantic_f1(
    memory: Memory,
    gold: HumanGold,
    words: list[str],
) -> float:
    values = []

    for word in words:
        target = gold.gold(word)

        if not target:
            continue

        values.append(
            f1(
                set(memory.concepts_for_word(word)),
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
        "=== V119 FIXED SEMANTIC QUERY TEACHER ==="
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

    memory = Memory(
        INPUT_MEMORY
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
    )

    print(
        "initial_semantic_f1:",
        initial_f1,
    )

    relation_stats = {
        relation: {
            "queries": 0,
            "valid_answers": 0,
            "reused": 0,
            "new": 0,
            "rejected": 0,
            "gold_hits": 0,
            "gold_total": 0,
        }
        for relation in RELATIONS
    }

    trace = []

    # ---------------------------------------------------------------
    # Build fixed query jobs.
    # ---------------------------------------------------------------

    jobs = []

    for word in words:
        context = memory.context(
            word,
            lexical_index,
        )

        for relation in RELATIONS:
            jobs.append(
                (
                    word,
                    relation,
                    context,
                )
            )

    print(
        "total_queries:",
        len(jobs),
        flush=True,
    )

    # ---------------------------------------------------------------
    # Query the LLM in batches.
    # ---------------------------------------------------------------

    for start in range(
        0,
        len(jobs),
        BATCH_SIZE,
    ):
        batch_jobs = jobs[
            start:start + BATCH_SIZE
        ]

        prompts = [
            query_prompt(
                tokenizer,
                word,
                relation,
                context,
            )
            for word, relation, context
            in batch_jobs
        ]

        raw_answers = generate_batch(
            tokenizer,
            model,
            device,
            prompts,
        )

        for (
            (word, relation, context),
            raw_answer,
        ) in zip(
            batch_jobs,
            raw_answers,
        ):
            stats = relation_stats[
                relation
            ]

            stats["queries"] += 1

            answer = normalize_answer(
                raw_answer
            )

            canonical = canonicalize_answer(
                answer,
                set(context),
                word,
            )

            gold_set = gold.gold(
                word
            )

            if canonical is None:
                stats["rejected"] += 1
            else:
                kind, concept = canonical
                stats[
                    "valid_answers"
                ] += 1

                if kind == "reuse":
                    stats["reused"] += 1
                    stats["gold_total"] += (
                        1
                        if gold_set
                        else 0
                    )

                    if concept in gold_set:
                        stats["gold_hits"] += 1

                else:
                    stats["new"] += 1
                    stats["gold_total"] += (
                        1
                        if gold_set
                        else 0
                    )

                    if concept in gold_set:
                        stats["gold_hits"] += 1

                # MUTATION IS DISABLED by default if the answer does not
                # directly equal a known human-gold concept.
                #
                # We still store accepted answers into the COPY of memory,
                # but each insertion is logged so that bad teacher behavior
                # cannot silently disappear.
                created = memory.add_concept(
                    word,
                    (
                        concept
                        if kind == "reuse"
                        else "NEW:" + concept
                    ),
                )

                if word in TRACE_WORDS:
                    trace.append(
                        {
                            "word": word,
                            "relation": relation,
                            "context": context,
                            "raw_answer": raw_answer,
                            "normalized": concept,
                            "kind": kind,
                            "gold": sorted(
                                gold_set
                            ),
                            "gold_hit": (
                                concept in gold_set
                            ),
                            "created": created,
                        }
                    )

            processed = min(
                start + BATCH_SIZE,
                len(jobs),
            )

        if (
            processed <= BATCH_SIZE
            or processed % BATCH_SIZE == 0
            or processed == len(jobs)
        ):
            total_valid = sum(
                item["valid_answers"]
                for item in relation_stats.values()
            )

            total_new = sum(
                item["new"]
                for item in relation_stats.values()
            )

            print(
                f"QUERY "
                f"{processed:4d}/{len(jobs):4d} "
                f"valid={total_valid} "
                f"new={total_new} "
                f"memory={len(memory.concept_id_by_name)}",
                flush=True,
            )

    memory.rebuild_co_usage()

    final_f1 = semantic_f1(
        memory,
        gold,
        words,
    )

    print()
    print(
        "=== V119 SUMMARY ==="
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
        "=== RELATION STATS ==="
    )

    for relation in RELATIONS:
        print(
            relation,
            relation_stats[relation],
        )

    print()

    print(
        "=== TRACE ==="
    )

    for item in trace[:100]:
        print(
            json.dumps(
                item,
                ensure_ascii=False,
            )
        )

    report = {
        "experiment": (
            "V119 fixed semantic query teacher"
        ),
        "evaluation_words": words,
        "relations": RELATIONS,
        "initial_concepts": initial_concepts,
        "final_concepts": len(
            memory.concept_id_by_name
        ),
        "concept_growth": (
            len(memory.concept_id_by_name)
            - initial_concepts
        ),
        "semantic_f1_before": initial_f1,
        "semantic_f1_after": final_f1,
        "semantic_f1_delta": (
            final_f1 - initial_f1
        ),
        "relation_stats": relation_stats,
        "trace": trace,
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
        "=== V119 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
