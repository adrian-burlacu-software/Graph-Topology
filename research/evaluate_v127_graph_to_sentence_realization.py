from __future__ import annotations

"""
V127 — GRAPH -> SENTENCE REALIZATION

Use the frozen SmolLM2-360M-Instruct as a language realizer, not as the
semantic teacher.

The graph supplies explicit ConceptNet facts. The LLM turns those facts into
one short English sentence.

Compression levels:
    100%, 10%, 5%, 2%, 1%

No graph mutation.
No semantic discovery.
"""

import csv
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "conceptnet_compact.db"
DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
MODEL_PATH = ROOT / "llm" / "SmolLM2-360M-Instruct"
OUTPUT_PATH = ROOT / "results" / "v127_graph_to_sentence_realization.json"

EVAL_WORDS = 256
BATCH_SIZE = 64
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 48
MAX_FACTS = 8

BUDGETS = (
    1.00,
    0.10,
    0.05,
    0.02,
    0.01,
)

TRACE_WORDS = {
    "hello",
    "greeting",
    "dog",
    "animal",
    "ability",
    "abandon",
    "water",
    "music",
    "chair",
    "car",
}

RELATION_PRIORITY = {
    "IsA": 4.0,
    "UsedFor": 4.0,
    "CapableOf": 3.8,
    "HasProperty": 3.6,
    "PartOf": 3.4,
    "HasA": 3.2,
    "Causes": 3.0,
    "AtLocation": 2.0,
    "MadeOf": 2.0,
    "HasPrerequisite": 2.0,
    "Synonym": 1.8,
    "Antonym": 1.5,
    "DefinedAs": 1.8,
    "RelatedTo": 1.0,
    "SimilarTo": 1.0,
    "HasContext": 0.8,
    "ReceivesAction": 1.5,
    "HasFirstSubevent": 1.5,
    "HasLastSubevent": 1.5,
    "MotivatedByGoal": 1.5,
}


def load_dictionary(path: Path) -> list[str]:
    words = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            word = raw.strip().lower()
            if word and word.isalpha():
                words.add(word)
    return sorted(words)[:EVAL_WORDS]


class ConceptNetDB:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()


class StudentGraph:
    def __init__(self) -> None:
        self.edges_by_start = defaultdict(list)
        self.edge_count = 0

    def add(self, row: sqlite3.Row) -> None:
        self.edges_by_start[row["start"]].append(
            (
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
        )
        self.edge_count += 1

    def facts(self, word: str, limit: int = MAX_FACTS):
        rows = self.edges_by_start.get(word, [])
        ranked = sorted(
            rows,
            key=lambda item: (
                -(
                    RELATION_PRIORITY.get(
                        item[0],
                        1.0,
                    )
                    * (
                        1.0
                        + 0.25
                        * min(
                            8.0,
                            item[2],
                        )
                    )
                ),
                item[0],
                item[1],
            ),
        )

        result = []
        seen = set()

        for relation, end, weight in ranked:
            key = (relation, end)
            if key in seen:
                continue
            seen.add(key)
            result.append((relation, end, weight))
            if len(result) >= limit:
                break

        return result


def load_teacher_rows(
    db: ConceptNetDB,
    words: set[str],
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in words)
    if not placeholders:
        return []

    return list(
        db.conn.execute(
            f"""
            SELECT start, relation, end, weight, dataset
            FROM edge
            WHERE start IN ({placeholders})
            """,
            tuple(words),
        )
    )


def endpoint_degree(rows):
    degree = {}
    for row in rows:
        degree[row["start"]] = degree.get(row["start"], 0) + 1
        degree[row["end"]] = degree.get(row["end"], 0) + 1
    return degree


def edge_score(row, degree) -> float:
    relation_weight = RELATION_PRIORITY.get(
        row["relation"],
        1.0,
    )

    weight = max(
        0.01,
        float(row["weight"]),
    )

    hub_penalty = 1.0 / (
        (
            1.0
            + degree.get(row["start"], 0)
            + degree.get(row["end"], 0)
        )
        ** 0.5
    )

    return (
        relation_weight
        * (1.0 + weight)
        * max(0.1, hub_penalty)
    )


def build_compressed_graphs(
    rows: list[sqlite3.Row],
    words: set[str],
):
    degree = endpoint_degree(rows)

    scored = [
        (
            edge_score(row, degree),
            row,
        )
        for row in rows
    ]

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1]["start"],
            item[1]["relation"],
            item[1]["end"],
        )
    )

    graphs = {}
    total = len(rows)

    for fraction in BUDGETS:
        graph = StudentGraph()
        budget = max(
            1,
            int(total * fraction),
        )

        selected = set()

        # Preserve at least one strong edge per dictionary word where budget
        # permits.
        best_for_word = {}
        for score, row in scored:
            word = row["start"]
            if word in words and word not in best_for_word:
                best_for_word[word] = (score, row)

        for score, row in sorted(
            best_for_word.values(),
            key=lambda item: -item[0],
        ):
            if graph.edge_count >= budget:
                break

            key = (
                row["start"],
                row["relation"],
                row["end"],
            )
            if key in selected:
                continue

            selected.add(key)
            graph.add(row)

        for score, row in scored:
            if graph.edge_count >= budget:
                break

            key = (
                row["start"],
                row["relation"],
                row["end"],
            )
            if key in selected:
                continue

            selected.add(key)
            graph.add(row)

        graphs[f"{fraction:.2f}"] = graph

    return graphs


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
            raise RuntimeError("Tokenizer has no PAD/EOS token.")
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
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    return tokenizer, model, device


def apply_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return system + "\n\n" + user + "\n\nAssistant:"


REALIZER_SYSTEM = (
    "You are a sentence realizer. "
    "The graph facts below are authoritative. "
    "Write one short natural English sentence about the target word. "
    "Use only facts supported by the graph. "
    "Do not add facts from your own knowledge. "
    "Return one sentence only."
)


def fact_to_text(word: str, relation: str, end: str) -> str:
    templates = {
        "IsA": f"{word} is a {end}",
        "PartOf": f"{word} is part of {end}",
        "HasA": f"{word} has {end}",
        "UsedFor": f"{word} is used for {end}",
        "CapableOf": f"{word} can {end}",
        "HasProperty": f"{word} has the property {end}",
        "Causes": f"{word} causes {end}",
        "AtLocation": f"{word} is found at {end}",
        "MadeOf": f"{word} is made of {end}",
        "ReceivesAction": f"{word} can be {end}",
        "HasPrerequisite": f"{word} requires {end}",
        "HasFirstSubevent": f"{word} first involves {end}",
        "HasLastSubevent": f"{word} ends with {end}",
        "MotivatedByGoal": f"{word} is motivated by {end}",
        "Synonym": f"{word} is similar in meaning to {end}",
        "Antonym": f"{word} is opposite to {end}",
        "RelatedTo": f"{word} is related to {end}",
        "SimilarTo": f"{word} is similar to {end}",
        "DefinedAs": f"{word} is defined as {end}",
        "HasContext": f"{word} is associated with {end}",
    }
    return templates.get(
        relation,
        f"{word} is related to {end}",
    )


def sentence_prompt(
    tokenizer,
    word: str,
    facts,
) -> str:
    fact_lines = [
        f"{i}. {fact_to_text(word, relation, end)}."
        for i, (relation, end, _weight) in enumerate(
            facts,
            start=1,
        )
    ]

    fact_text = "\n".join(fact_lines)

    return apply_chat(
        tokenizer,
        REALIZER_SYSTEM,
        f"""
TARGET:
{word}

GRAPH FACTS:
{fact_text}

Write ONE simple sentence about TARGET using the graph facts.

Rules:
- Keep TARGET as the subject when natural.
- Combine compatible facts when possible.
- Do not mention graph, facts, ConceptNet, candidates, or instructions.
- Do not invent details.
- One sentence only.
""".strip(),
    )


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
                output[row, prompt_len:],
                skip_special_tokens=True,
            ).strip()
        )

    return results


def clean_sentence(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^```.*?$",
        "",
        text,
        flags=re.M,
    ).strip()

    text = re.sub(
        r"^(answer|response|sentence)\s*:\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    match = re.search(
        r"(.+?[.!?])(?:\s|$)",
        text,
    )

    if match:
        text = match.group(1).strip()

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def sentence_quality(
    word: str,
    sentence: str,
) -> dict[str, object]:
    lower = sentence.lower()

    has_target = word.lower() in lower

    instruction_leak = any(
        marker in lower
        for marker in (
            "graph facts",
            "candidate",
            "conceptnet",
            "do not",
            "return one",
            "assistant",
        )
    )

    sentence_count = len(
        re.findall(
            r"[.!?]+",
            sentence,
        )
    )

    clean = (
        bool(sentence)
        and has_target
        and not instruction_leak
        and sentence_count <= 2
        and len(sentence) <= 240
    )

    return {
        "has_target": has_target,
        "instruction_leak": instruction_leak,
        "sentence_count": sentence_count,
        "chars": len(sentence),
        "clean": clean,
    }


def fact_overlap(
    sentence: str,
    facts,
) -> int:
    lower = sentence.lower()
    hits = 0

    for _relation, end, _weight in facts:
        tokens = [
            token
            for token in re.findall(
                r"[a-z0-9]+",
                end.lower(),
            )
            if len(token) >= 3
        ]

        if tokens and any(
            token in lower
            for token in tokens
        ):
            hits += 1

    return hits


def main() -> None:
    started = time.perf_counter()

    print(
        "=== V127 GRAPH -> SENTENCE REALIZATION ==="
    )

    words = load_dictionary(DICTIONARY_PATH)
    word_set = set(words)

    db = ConceptNetDB(DB_PATH)

    try:
        print(
            "dictionary_words:",
            len(words),
            flush=True,
        )

        print(
            "loading dictionary-centered ConceptNet rows...",
            flush=True,
        )

        rows = load_teacher_rows(
            db,
            word_set,
        )

        print(
            "teacher_rows:",
            f"{len(rows):,}",
            flush=True,
        )

        graphs = build_compressed_graphs(
            rows,
            word_set,
        )

        for name, graph in graphs.items():
            print(
                f"graph {name}: edges={graph.edge_count}",
                flush=True,
            )

        tokenizer, model, device = load_model()

        all_results = {}

        for budget_name, graph in graphs.items():
            print(
                f"\n=== GRAPH {budget_name} ===",
                flush=True,
            )

            facts_by_word = {
                word: graph.facts(
                    word,
                    limit=MAX_FACTS,
                )
                for word in words
            }

            jobs = [
                (word, facts_by_word[word])
                for word in words
                if facts_by_word[word]
            ]

            outputs = {}
            quality = {}
            overlaps = []

            generation_started = time.perf_counter()

            for start in range(
                0,
                len(jobs),
                BATCH_SIZE,
            ):
                batch = jobs[
                    start:start + BATCH_SIZE
                ]

                prompts = [
                    sentence_prompt(
                        tokenizer,
                        word,
                        facts,
                    )
                    for word, facts in batch
                ]

                raw_outputs = generate_batch(
                    tokenizer,
                    model,
                    device,
                    prompts,
                )

                for (
                    (word, facts),
                    raw,
                ) in zip(
                    batch,
                    raw_outputs,
                ):
                    sentence = clean_sentence(raw)

                    outputs[word] = sentence

                    quality[word] = sentence_quality(
                        word,
                        sentence,
                    )

                    overlaps.append(
                        fact_overlap(
                            sentence,
                            facts,
                        )
                    )

                    if word in TRACE_WORDS:
                        print(
                            f"\n[{budget_name}] {word}",
                            flush=True,
                        )
                        print(
                            "facts:",
                            facts,
                            flush=True,
                        )
                        print(
                            "sentence:",
                            sentence,
                            flush=True,
                        )
                        print(
                            "quality:",
                            quality[word],
                            flush=True,
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
                    elapsed = (
                        time.perf_counter()
                        - generation_started
                    )

                    print(
                        f"REALIZE "
                        f"{processed:4d}/{len(jobs):4d} "
                        f"sent/s="
                        f"{processed / max(0.001, elapsed):.2f}",
                        flush=True,
                    )

            generation_seconds = (
                time.perf_counter()
                - generation_started
            )

            clean_count = sum(
                bool(item["clean"])
                for item in quality.values()
            )

            target_count = sum(
                bool(item["has_target"])
                for item in quality.values()
            )

            leak_count = sum(
                bool(item["instruction_leak"])
                for item in quality.values()
            )

            result = {
                "graph_edges": graph.edge_count,
                "words_with_facts": len(jobs),
                "generation_seconds": generation_seconds,
                "sentences_per_second": (
                    len(jobs)
                    / max(
                        0.001,
                        generation_seconds,
                    )
                ),
                "target_presence_rate": (
                    target_count
                    / max(1, len(jobs))
                ),
                "clean_sentence_rate": (
                    clean_count
                    / max(1, len(jobs))
                ),
                "instruction_leak_rate": (
                    leak_count
                    / max(1, len(jobs))
                ),
                "mean_fact_overlap": (
                    sum(overlaps)
                    / max(
                        1,
                        len(overlaps),
                    )
                ),
                "sentences": outputs,
                "quality": quality,
                "facts": {
                    word: [
                        {
                            "relation": relation,
                            "end": end,
                            "weight": weight,
                        }
                        for relation, end, weight in facts
                    ]
                    for word, facts in facts_by_word.items()
                },
            }

            all_results[budget_name] = result

        print(
            "\n=== V127 REALIZATION CURVE ==="
        )
        print(
            "budget | edges | sent/s | clean | target | overlap"
        )

        for budget_name, result in all_results.items():
            print(
                f"{budget_name:6s} | "
                f"{result['graph_edges']:7d} | "
                f"{result['sentences_per_second']:.2f} | "
                f"{result['clean_sentence_rate']:.4f} | "
                f"{result['target_presence_rate']:.4f} | "
                f"{result['mean_fact_overlap']:.2f}"
            )

        report = {
            "experiment": "V127 graph to sentence realization",
            "dictionary_words": len(words),
            "teacher_rows": len(rows),
            "budgets": BUDGETS,
            "results": all_results,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                report,
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
            "=== V127 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
