from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LLM = ROOT / "llm"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_DIR = LLM / "SmolLM2-1.7B-Instruct"

REPORT_PATH = RESULTS / "v204_teacher_qualification.json"
RESPONSES_PATH = RESULTS / "v204_teacher_responses.jsonl"


# ---------------------------------------------------------------------------
# Relations and surface forms
# ---------------------------------------------------------------------------

RELATIONS = (
    "IsA",
    "CapableOf",
    "HasProperty",
    "UsedFor",
    "HasA",
    "PartOf",
    "RelatedTo",
    "SimilarTo",
    "Antonym",
    "Causes",
    "AtLocation",
)

RELATION_TEMPLATES = {
    "IsA": (
        "A {source} is a kind of {target}.",
        "{source} is a type of {target}.",
        "{source} belongs to the category {target}.",
    ),
    "CapableOf": (
        "{source} can {target}.",
        "{source} is capable of {target}.",
        "{source} is able to {target}.",
    ),
    "HasProperty": (
        "{source} is {target}.",
        "{source} has the property {target}.",
        "{source} can be described as {target}.",
    ),
    "UsedFor": (
        "{source} is used for {target}.",
        "The purpose of {source} is {target}.",
        "People use {source} for {target}.",
    ),
    "HasA": (
        "{source} has a {target}.",
        "{source} contains a {target}.",
        "A {source} can have a {target}.",
    ),
    "PartOf": (
        "A {source} is part of a {target}.",
        "{source} is a component of {target}.",
        "{source} belongs to a larger {target}.",
    ),
    "RelatedTo": (
        "{source} is related to {target}.",
        "{source} is associated with {target}.",
        "{source} is connected with {target}.",
    ),
    "SimilarTo": (
        "{source} is similar to {target}.",
        "{source} resembles {target}.",
        "{source} is much like {target}.",
    ),
    "Antonym": (
        "{source} is the opposite of {target}.",
        "{source} contrasts with {target}.",
        "{source} and {target} are opposites.",
    ),
    "Causes": (
        "{source} causes {target}.",
        "{target} can result from {source}.",
        "{source} can lead to {target}.",
    ),
    "AtLocation": (
        "{source} is found in {target}.",
        "{source} can be located at {target}.",
        "You may find {source} in {target}.",
    ),
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a semantic graph parser.

Convert the user's sentence into a minimal semantic graph.

Return ONLY valid JSON with exactly this shape:
{
  "nodes": ["..."],
  "edges": [
    {"source": "...", "relation": "...", "target": "..."}
  ]
}

Allowed relations:
IsA, CapableOf, HasProperty, UsedFor, HasA, PartOf,
RelatedTo, SimilarTo, Antonym, Causes, AtLocation

Rules:
- Use lowercase concept strings.
- Do not invent extra concepts.
- Only include relations directly supported by the sentence.
- Do not explain your answer.
- Do not wrap JSON in markdown fences.
"""

USER_TEMPLATE = """Parse this sentence into the semantic graph:

{sentence}
"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(
        r"\s+",
        " ",
        text,
    )
    text = text.strip(" .,!?:;\"'")
    return text


def normalize_relation(
    relation: Any,
) -> str | None:
    if not isinstance(relation, str):
        return None

    relation = relation.strip()

    aliases = {
        "isa": "IsA",
        "is_a": "IsA",
        "is a": "IsA",
        "capableof": "CapableOf",
        "capable_of": "CapableOf",
        "hasproperty": "HasProperty",
        "has_property": "HasProperty",
        "usedfor": "UsedFor",
        "used_for": "UsedFor",
        "hasa": "HasA",
        "has_a": "HasA",
        "partof": "PartOf",
        "part_of": "PartOf",
        "relatedto": "RelatedTo",
        "related_to": "RelatedTo",
        "similar_to": "SimilarTo",
        "similarto": "SimilarTo",
        "antonym": "Antonym",
        "causes": "Causes",
        "atlocation": "AtLocation",
        "at_location": "AtLocation",
    }

    if relation in RELATIONS:
        return relation

    return aliases.get(
        relation.lower()
    )


def extract_json_object(
    text: str,
) -> dict | None:
    text = text.strip()

    # First try the complete text.
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Then extract the first balanced JSON object.
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(
        start,
        len(text),
    ):
        char = text[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[
                    start : i + 1
                ]
                try:
                    value = json.loads(
                        candidate
                    )
                    if isinstance(
                        value,
                        dict,
                    ):
                        return value
                except json.JSONDecodeError:
                    return None

    return None


def parse_graph(
    text: str,
) -> tuple[set[str], set[tuple[str, str, str]], str | None]:
    parsed = extract_json_object(
        text
    )

    if parsed is None:
        return set(), set(), "json_parse"

    raw_nodes = parsed.get(
        "nodes",
        [],
    )
    raw_edges = parsed.get(
        "edges",
        [],
    )

    nodes: set[str] = set()

    if isinstance(
        raw_nodes,
        list,
    ):
        for node in raw_nodes:
            if isinstance(node, str):
                nodes.add(
                    normalize_text(node)
                )

    edges: set[tuple[str, str, str]] = set()

    if isinstance(
        raw_edges,
        list,
    ):
        for edge in raw_edges:
            if not isinstance(
                edge,
                dict,
            ):
                continue

            source = edge.get(
                "source"
            )
            relation = normalize_relation(
                edge.get("relation")
            )
            target = edge.get(
                "target"
            )

            if (
                isinstance(source, str)
                and relation is not None
                and isinstance(target, str)
            ):
                edges.add(
                    (
                        normalize_text(source),
                        relation,
                        normalize_text(target),
                    )
                )

    # Edges imply nodes even if the model omitted them from "nodes".
    for source, _relation, target in edges:
        nodes.add(source)
        nodes.add(target)

    return nodes, edges, None


# ---------------------------------------------------------------------------
# Database sampling
# ---------------------------------------------------------------------------

def load_examples(
    *,
    single_facts: int,
    multi_facts: int,
    seed: int = 204,
) -> list[dict]:
    conn = sqlite3.connect(
        str(DB_PATH)
    )
    conn.row_factory = sqlite3.Row

    try:
        per_relation = max(
            1,
            single_facts
            // len(RELATIONS),
        )

        rows = conn.execute(
            """
            SELECT start, relation, end, weight
            FROM edge
            WHERE relation IN (
                'IsA',
                'CapableOf',
                'HasProperty',
                'UsedFor',
                'HasA',
                'PartOf',
                'RelatedTo',
                'SimilarTo',
                'Antonym',
                'Causes',
                'AtLocation'
            )
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (
                max(
                    single_facts * 4,
                    500,
                ),
            ),
        ).fetchall()
    finally:
        conn.close()

    examples = []
    relation_counts = Counter()
    used_pairs = set()

    for row in rows:
        relation = row["relation"]

        if relation_counts[
            relation
        ] >= per_relation:
            continue

        source = normalize_text(
            row["start"]
        )
        target = normalize_text(
            row["end"]
        )

        key = (
            source,
            relation,
            target,
        )

        if key in used_pairs:
            continue

        used_pairs.add(key)

        templates = RELATION_TEMPLATES[
            relation
        ]

        template = templates[
            relation_counts[relation]
            % len(templates)
        ]

        sentence = template.format(
            source=source,
            target=target,
        )

        examples.append(
            {
                "kind": "single",
                "sentence": sentence,
                "expected_nodes": [
                    source,
                    target,
                ],
                "expected_edges": [
                    (
                        source,
                        relation,
                        target,
                    )
                ],
            }
        )

        relation_counts[
            relation
        ] += 1

        if len(examples) >= single_facts:
            break

    # Multi-fact examples are built from two facts sharing the subject where
    # possible. If no pair is found, use adjacent facts from the single set.
    by_source: dict[str, list[dict]] = {}

    for item in examples:
        edge = item["expected_edges"][0]
        by_source.setdefault(
            edge[0],
            [],
        ).append(
            item
        )

    multi_candidates = [
        items
        for items in by_source.values()
        if len(items) >= 2
    ]

    rng = __import__(
        "random"
    ).Random(seed)

    rng.shuffle(
        multi_candidates
    )

    for items in multi_candidates:
        if len(
            examples
        ) >= single_facts + multi_facts:
            break

        a = items[0]
        b = items[1]

        sentence = (
            a["sentence"].rstrip(".")
            + " "
            + b["sentence"]
        )

        nodes = sorted(
            set(
                a["expected_nodes"]
                + b["expected_nodes"]
            )
        )
        edges = (
            a["expected_edges"]
            + b["expected_edges"]
        )

        examples.append(
            {
                "kind": "multi",
                "sentence": sentence,
                "expected_nodes": nodes,
                "expected_edges": edges,
            }
        )

    return examples[
        : single_facts + multi_facts
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(
    predicted: set,
    expected: set,
) -> tuple[float, float, float]:
    if not predicted:
        precision = 1.0 if not expected else 0.0
    else:
        precision = len(
            predicted & expected
        ) / len(predicted)

    recall = (
        1.0
        if not expected and not predicted
        else len(
            predicted & expected
        )
        / max(
            1,
            len(expected),
        )
    )

    if (
        precision + recall
    ) == 0.0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return precision, recall, f1


def edge_relation_accuracy(
    predicted: set[tuple[str, str, str]],
    expected: set[tuple[str, str, str]],
) -> float:
    if not expected:
        return 1.0

    correct = 0

    for source, expected_relation, target in expected:
        if (
            source,
            expected_relation,
            target,
        ) in predicted:
            correct += 1
            continue

        # Same endpoints, wrong relation.
        endpoint_match = any(
            ps == source
            and pt == target
            for ps, _pr, pt in predicted
        )

        if endpoint_match:
            continue

    return (
        correct
        / len(expected)
    )


# ---------------------------------------------------------------------------
# Model loading / generation
# ---------------------------------------------------------------------------

def load_model(
    device: torch.device,
):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"SmolLM2 model directory not found: "
            f"{MODEL_DIR.resolve()}"
        )

    print(
        "model_dir:",
        MODEL_DIR.resolve(),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer, model


def generate(
    tokenizer,
    model,
    device: torch.device,
    sentence: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    user_prompt = USER_TEMPLATE.format(
        sentence=sentence
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    if hasattr(
        tokenizer,
        "apply_chat_template",
    ):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + user_prompt
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(
                temperature > 0.0
            ),
            temperature=(
                temperature
                if temperature > 0.0
                else None
            ),
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = output[
        0
    ][
        inputs["input_ids"].shape[1] :
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


# ---------------------------------------------------------------------------
# Main qualification run
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-facts",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--multi-facts",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== V204 SMOLLM2 TEACHER QUALIFICATION ===",
        flush=True,
    )
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

    print(
        "conceptnet_db:",
        DB_PATH.resolve(),
        "exists=",
        DB_PATH.exists(),
        flush=True,
    )
    print(
        "model_dir:",
        MODEL_DIR.resolve(),
        "exists=",
        MODEL_DIR.exists(),
        flush=True,
    )

    if not DB_PATH.exists():
        raise FileNotFoundError(
            DB_PATH.resolve()
        )

    tokenizer, model = load_model(
        device
    )

    examples = load_examples(
        single_facts=args.single_facts,
        multi_facts=args.multi_facts,
    )[
        : args.limit
    ]

    print(
        "cases:",
        len(examples),
        flush=True,
    )

    total = len(examples)

    report_items = []

    node_tp = 0
    node_fp = 0
    node_fn = 0

    edge_tp = 0
    edge_fp = 0
    edge_fn = 0

    json_success = 0
    relation_correct = 0

    per_kind = {
        "single": [],
        "multi": [],
    }

    start_time = time.perf_counter()

    with RESPONSES_PATH.open(
        "w",
        encoding="utf-8",
    ) as response_file:

        for index, example in enumerate(
            examples,
            start=1,
        ):
            raw = generate(
                tokenizer,
                model,
                device,
                example["sentence"],
                args.max_new_tokens,
                args.temperature,
            )

            (
                predicted_nodes,
                predicted_edges,
                parse_error,
            ) = parse_graph(
                raw
            )

            expected_nodes = set(
                example[
                    "expected_nodes"
                ]
            )

            expected_edges = set(
                example[
                    "expected_edges"
                ]
            )

            if parse_error is None:
                json_success += 1

            predicted_node_tp = len(
                predicted_nodes
                & expected_nodes
            )
            predicted_node_fp = len(
                predicted_nodes
                - expected_nodes
            )
            predicted_node_fn = len(
                expected_nodes
                - predicted_nodes
            )

            predicted_edge_tp = len(
                predicted_edges
                & expected_edges
            )
            predicted_edge_fp = len(
                predicted_edges
                - expected_edges
            )
            predicted_edge_fn = len(
                expected_edges
                - predicted_edges
            )

            node_tp += predicted_node_tp
            node_fp += predicted_node_fp
            node_fn += predicted_node_fn

            edge_tp += predicted_edge_tp
            edge_fp += predicted_edge_fp
            edge_fn += predicted_edge_fn

            relation_acc = (
                edge_relation_accuracy(
                    predicted_edges,
                    expected_edges,
                )
            )

            relation_correct += relation_acc

            item = {
                "index": index,
                "kind": example["kind"],
                "sentence": example["sentence"],
                "expected": {
                    "nodes": sorted(
                        expected_nodes
                    ),
                    "edges": [
                        {
                            "source": s,
                            "relation": r,
                            "target": t,
                        }
                        for s, r, t
                        in sorted(
                            expected_edges
                        )
                    ],
                },
                "raw_output": raw,
                "predicted": {
                    "nodes": sorted(
                        predicted_nodes
                    ),
                    "edges": [
                        {
                            "source": s,
                            "relation": r,
                            "target": t,
                        }
                        for s, r, t
                        in sorted(
                            predicted_edges
                        )
                    ],
                },
                "parse_error": parse_error,
                "node_tp": predicted_node_tp,
                "node_fp": predicted_node_fp,
                "node_fn": predicted_node_fn,
                "edge_tp": predicted_edge_tp,
                "edge_fp": predicted_edge_fp,
                "edge_fn": predicted_edge_fn,
                "relation_accuracy": relation_acc,
            }

            per_kind[
                example["kind"]
            ].append(item)

            report_items.append(item)

            response_file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )
            response_file.flush()

            if (
                index <= 5
                or index % 10 == 0
                or index == total
            ):
                print(
                    f"CASE {index:3d}/{total:3d} "
                    f"parse={'OK' if parse_error is None else 'FAIL'} "
                    f"edges={len(predicted_edges):2d} "
                    f"edge_hit={predicted_edge_tp:1d}",
                    flush=True,
                )

    node_precision = (
        node_tp
        / max(
            1,
            node_tp + node_fp,
        )
    )
    node_recall = (
        node_tp
        / max(
            1,
            node_tp + node_fn,
        )
    )
    node_f1 = (
        2
        * node_precision
        * node_recall
        / max(
            1e-12,
            node_precision + node_recall,
        )
    )

    edge_precision = (
        edge_tp
        / max(
            1,
            edge_tp + edge_fp,
        )
    )
    edge_recall = (
        edge_tp
        / max(
            1,
            edge_tp + edge_fn,
        )
    )
    edge_f1 = (
        2
        * edge_precision
        * edge_recall
        / max(
            1e-12,
            edge_precision + edge_recall,
        )
    )

    hallucinated_edge_rate = (
        edge_fp
        / max(
            1,
            edge_tp + edge_fp,
        )
    )

    missed_edge_rate = (
        edge_fn
        / max(
            1,
            edge_tp + edge_fn,
        )
    )

    by_kind_summary = {}

    for kind, items in per_kind.items():
        if not items:
            continue

        kind_edge_tp = sum(
            item["edge_tp"]
            for item in items
        )
        kind_edge_fp = sum(
            item["edge_fp"]
            for item in items
        )
        kind_edge_fn = sum(
            item["edge_fn"]
            for item in items
        )

        kind_precision = (
            kind_edge_tp
            / max(
                1,
                kind_edge_tp
                + kind_edge_fp,
            )
        )
        kind_recall = (
            kind_edge_tp
            / max(
                1,
                kind_edge_tp
                + kind_edge_fn,
            )
        )
        kind_f1 = (
            2
            * kind_precision
            * kind_recall
            / max(
                1e-12,
                kind_precision
                + kind_recall,
            )
        )

        by_kind_summary[
            kind
        ] = {
            "cases": len(items),
            "parse_success": (
                sum(
                    item["parse_error"]
                    is None
                    for item in items
                )
                / len(items)
            ),
            "edge_precision": kind_precision,
            "edge_recall": kind_recall,
            "edge_f1": kind_f1,
            "relation_accuracy": (
                sum(
                    item["relation_accuracy"]
                    for item in items
                )
                / len(items)
            ),
        }

    elapsed = (
        time.perf_counter()
        - start_time
    )

    report = {
        "experiment": (
            "V204 SmolLM2 teacher qualification"
        ),
        "model_dir": str(
            MODEL_DIR.resolve()
        ),
        "device": str(device),
        "cases": total,
        "single_cases": args.single_facts,
        "multi_cases": args.multi_facts,
        "json_parse_success": (
            json_success
            / max(
                1,
                total,
            )
        ),
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "relation_accuracy": (
            relation_correct
            / max(
                1,
                total,
            )
        ),
        "hallucinated_edge_rate": hallucinated_edge_rate,
        "missed_edge_rate": missed_edge_rate,
        "by_kind": by_kind_summary,
        "responses_path": str(
            RESPONSES_PATH.resolve()
        ),
        "elapsed_seconds": elapsed,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== V204 SUMMARY ==="
    )
    print(
        "json_parse_success:",
        f"{report['json_parse_success']:.4f}",
    )
    print(
        "node_f1:",
        f"{node_f1:.4f}",
    )
    print(
        "edge_precision:",
        f"{edge_precision:.4f}",
    )
    print(
        "edge_recall:",
        f"{edge_recall:.4f}",
    )
    print(
        "edge_f1:",
        f"{edge_f1:.4f}",
    )
    print(
        "relation_accuracy:",
        f"{report['relation_accuracy']:.4f}",
    )
    print(
        "hallucinated_edge_rate:",
        f"{hallucinated_edge_rate:.4f}",
    )
    print(
        "missed_edge_rate:",
        f"{missed_edge_rate:.4f}",
    )

    print()
    print(
        "by_kind:"
    )
    for kind, metrics in by_kind_summary.items():
        print(
            f"  {kind:6s} "
            f"parse={metrics['parse_success']:.4f} "
            f"edge_f1={metrics['edge_f1']:.4f} "
            f"relation={metrics['relation_accuracy']:.4f}"
        )

    print()
    print(
        "report:",
        REPORT_PATH.resolve(),
    )
    print(
        "raw responses:",
        RESPONSES_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
