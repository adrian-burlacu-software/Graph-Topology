from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sqlite3
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LLM = ROOT / "llm"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_DIR = LLM / "SmolLM2-1.7B-Instruct"

REPORT_PATH = RESULTS / "v204b_teacher_qualification.json"
RESPONSES_PATH = RESULTS / "v204b_teacher_responses.jsonl"

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

# Curated semantic templates. We deliberately avoid impossible English
# constructions for relations like Causes and AtLocation.
TEMPLATES = {
    "IsA": [
        "A {source} is a kind of {target}.",
        "{source} is a type of {target}.",
        "A {source} belongs to the category {target}.",
    ],
    "CapableOf": [
        "A {source} can {target}.",
        "{source} is capable of {target}.",
        "{source} is able to {target}.",
    ],
    "HasProperty": [
        "{source} is {target}.",
        "{source} has the property of being {target}.",
        "{source} can be described as {target}.",
    ],
    "UsedFor": [
        "{source} is used for {target}.",
        "People use {source} for {target}.",
        "The purpose of {source} is {target}.",
    ],
    "HasA": [
        "A {source} has a {target}.",
        "{source} contains a {target}.",
        "{source} can have a {target}.",
    ],
    "PartOf": [
        "A {source} is part of a {target}.",
        "{source} is a component of {target}.",
        "{source} belongs to a larger {target}.",
    ],
    "RelatedTo": [
        "{source} is related to {target}.",
        "{source} is associated with {target}.",
        "{source} is connected with {target}.",
    ],
    "SimilarTo": [
        "{source} is similar to {target}.",
        "{source} resembles {target}.",
        "{source} is much like {target}.",
    ],
    "Antonym": [
        "{source} is the opposite of {target}.",
        "{source} and {target} are opposites.",
        "{source} contrasts with {target}.",
    ],
    "Causes": [
        "{source} causes {target}.",
        "{source} can lead to {target}.",
        "{source} can result in {target}.",
    ],
    "AtLocation": [
        "{source} is found in {target}.",
        "{source} can be found at {target}.",
        "You may find {source} in {target}.",
    ],
}

RELATION_ALIASES = {
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
    "similarto": "SimilarTo",
    "similar_to": "SimilarTo",
    "antonym": "Antonym",
    "causes": "Causes",
    "atlocation": "AtLocation",
    "at_location": "AtLocation",
}

SYSTEM_PROMPT = """You are a semantic parser.

You will receive a short natural-language statement.

You must extract only the semantic facts explicitly stated by the text.
Do not infer extra facts.

Allowed relation names:
IsA, CapableOf, HasProperty, UsedFor, HasA, PartOf,
RelatedTo, SimilarTo, Antonym, Causes, AtLocation

Return ONLY the requested JSON object.
Do not use markdown fences.
Use lowercase concepts exactly as they appear, normalized to simple nouns or
noun phrases.
"""

GRAPH_EXAMPLES = [
    (
        "A dog is an animal.",
        '{"nodes":["dog","animal"],"edges":[{"source":"dog","relation":"IsA","target":"animal"}]}',
    ),
    (
        "A dog can bark.",
        '{"nodes":["dog","bark"],"edges":[{"source":"dog","relation":"CapableOf","target":"bark"}]}',
    ),
]

CLASSIFICATION_EXAMPLES = [
    (
        "A dog is an animal.",
        '{"relation":"IsA"}',
    ),
    (
        "A dog can bark.",
        '{"relation":"CapableOf"}',
    ),
]

TRAJECTORY_EXAMPLE = (
    """Current state:
- dog is active
- animal is present but inactive

Next state:
- dog is active
- animal is active
- dog is bound to animal with relation IsA
""",
    '{"current_nodes":["dog","animal"],"next_nodes":["dog","animal"],"edges_added":[{"source":"dog","relation":"IsA","target":"animal"}]}',
)


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .,!?:;\"'")
    return value


def relation_from_value(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value in RELATIONS:
        return value
    return RELATION_ALIASES.get(value.lower())


def parse_json_object(text: str) -> dict | None:
    text = text.strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Search for the first balanced JSON object. This handles chat models
    # that prepend a tiny amount of text despite the instruction.
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    return result if isinstance(result, dict) else None
                except json.JSONDecodeError:
                    return None

    return None


def parse_graph(
    text: str,
) -> tuple[set[str], set[tuple[str, str, str]], str | None]:
    obj = parse_json_object(text)
    if obj is None:
        return set(), set(), "json"

    nodes = set()
    raw_nodes = obj.get("nodes", [])
    if isinstance(raw_nodes, list):
        for node in raw_nodes:
            if isinstance(node, str):
                nodes.add(normalize(node))

    edges = set()
    raw_edges = obj.get("edges", [])
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue

            source = edge.get("source")
            target = edge.get("target")
            relation = relation_from_value(
                edge.get("relation")
            )

            if (
                isinstance(source, str)
                and isinstance(target, str)
                and relation is not None
            ):
                edges.add(
                    (
                        normalize(source),
                        relation,
                        normalize(target),
                    )
                )

    for source, _relation, target in edges:
        nodes.add(source)
        nodes.add(target)

    return nodes, edges, None


def parse_relation(
    text: str,
) -> str | None:
    obj = parse_json_object(text)
    if obj is None:
        return None
    return relation_from_value(
        obj.get("relation")
    )


def parse_trajectory(
    text: str,
) -> tuple[set[str], set[str], set[tuple[str, str, str]], str | None]:
    obj = parse_json_object(text)
    if obj is None:
        return set(), set(), set(), "json"

    current_nodes = {
        normalize(x)
        for x in obj.get(
            "current_nodes",
            [],
        )
        if isinstance(x, str)
    }

    next_nodes = {
        normalize(x)
        for x in obj.get(
            "next_nodes",
            [],
        )
        if isinstance(x, str)
    }

    edges = set()

    for edge in obj.get(
        "edges_added",
        [],
    ):
        if not isinstance(edge, dict):
            continue

        source = edge.get("source")
        target = edge.get("target")
        relation = relation_from_value(
            edge.get("relation")
        )

        if (
            isinstance(source, str)
            and isinstance(target, str)
            and relation is not None
        ):
            edges.add(
                (
                    normalize(source),
                    relation,
                    normalize(target),
                )
            )

    return (
        current_nodes,
        next_nodes,
        edges,
        None,
    )


def balanced_f1(
    predicted: set,
    expected: set,
) -> tuple[float, float, float]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    precision = (
        tp / max(1, tp + fp)
    )
    recall = (
        tp / max(1, tp + fn)
    )
    f1 = (
        2 * precision * recall
        / max(
            1e-12,
            precision + recall,
        )
    )

    return precision, recall, f1


def sample_edges(
    count: int,
) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(
        str(DB_PATH)
    )
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT start, relation, end
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
              AND LENGTH(start) <= 40
              AND LENGTH(end) <= 40
            ORDER BY RANDOM()
            LIMIT 10000
            """
        ).fetchall()
    finally:
        conn.close()

    rng = random.Random(204)
    rng.shuffle(rows)

    selected = []
    per_relation = defaultdict(int)
    quota = max(
        1,
        count // len(RELATIONS),
    )

    for row in rows:
        relation = row["relation"]
        if per_relation[relation] >= quota:
            continue

        selected.append(
            (
                normalize(row["start"]),
                relation,
                normalize(row["end"]),
            )
        )
        per_relation[relation] += 1

        if len(selected) >= count:
            break

    # If balancing quotas left us short, fill from remaining rows.
    if len(selected) < count:
        seen = set(selected)
        for row in rows:
            item = (
                normalize(row["start"]),
                row["relation"],
                normalize(row["end"]),
            )
            if item in seen:
                continue
            selected.append(item)
            seen.add(item)
            if len(selected) >= count:
                break

    if len(selected) < count:
        raise RuntimeError(
            f"Could only sample {len(selected)} graph facts; requested {count}."
        )

    return selected


def build_cases(
    single: int = 50,
    multi: int = 30,
    trajectory: int = 20,
) -> list[dict]:
    total_edges_needed = max(
        single + multi * 2 + trajectory,
        200,
    )

    edges = sample_edges(
        total_edges_needed
    )

    cases = []

    # A/B: same facts are reused where sensible, so relation vs graph
    # extraction can be compared directly.
    for index, (
        source,
        relation,
        target,
    ) in enumerate(
        edges[:single],
        1,
    ):
        template = TEMPLATES[
            relation
        ][
            index
            % len(
                TEMPLATES[relation]
            )
        ]

        sentence = template.format(
            source=source,
            target=target,
        )

        cases.append(
            {
                "id": f"relation_{index:03d}",
                "kind": "relation",
                "sentence": sentence,
                "expected_relation": relation,
            }
        )

    for index, (
        source,
        relation,
        target,
    ) in enumerate(
        edges[single : single * 2],
        1,
    ):
        template = TEMPLATES[
            relation
        ][
            index
            % len(
                TEMPLATES[relation]
            )
        ]

        sentence = template.format(
            source=source,
            target=target,
        )

        cases.append(
            {
                "id": f"single_{index:03d}",
                "kind": "single",
                "sentence": sentence,
                "expected_nodes": {
                    source,
                    target,
                },
                "expected_edges": {
                    (
                        source,
                        relation,
                        target,
                    )
                },
            }
        )

    # Multi-fact examples are built explicitly from two facts with the same
    # source when possible, otherwise two independent facts.
    multi_source_pool = edges[
        single * 2 :
    ]

    groups = defaultdict(list)
    for edge in multi_source_pool:
        groups[edge[0]].append(edge)

    pairs = []
    for group in groups.values():
        if len(group) >= 2:
            pairs.append(
                (
                    group[0],
                    group[1],
                )
            )
        if len(pairs) >= multi:
            break

    if len(pairs) < multi:
        for i in range(
            0,
            min(
                len(multi_source_pool) - 1,
                multi * 2,
            ),
            2,
        ):
            pair = (
                multi_source_pool[i],
                multi_source_pool[i + 1],
            )
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= multi:
                break

    if len(pairs) < multi:
        raise RuntimeError(
            f"Could only construct {len(pairs)} multi-fact cases; requested {multi}."
        )

    for index, pair in enumerate(
        pairs[:multi],
        1,
    ):
        sentences = []
        expected_nodes = set()
        expected_edges = set()

        for source, relation, target in pair:
            template = TEMPLATES[
                relation
            ][
                index
                % len(
                    TEMPLATES[relation]
                )
            ]

            sentences.append(
                template.format(
                    source=source,
                    target=target,
                )
            )
            expected_nodes.update(
                (
                    source,
                    target,
                )
            )
            expected_edges.add(
                (
                    source,
                    relation,
                    target,
                )
            )

        cases.append(
            {
                "id": f"multi_{index:03d}",
                "kind": "multi",
                "sentence": " ".join(
                    sentences
                ),
                "expected_nodes": expected_nodes,
                "expected_edges": expected_edges,
            }
        )

    # Trajectory cases use explicit graph state language. This is the important
    # pre-teacher test for the future V205 trajectory teacher.
    for index, (
        source,
        relation,
        target,
    ) in enumerate(
        edges[
            single * 2 + multi * 2 :
            single * 2 + multi * 2 + trajectory
        ],
        1,
    ):
        sentence = (
            f"Current state: {source} is active. "
            f"{target} is present but inactive. "
            f"Next state: {source} is active and {target} is active. "
            f"The next state binds {source} to {target} "
            f"with relation {relation}."
        )

        cases.append(
            {
                "id": f"trajectory_{index:03d}",
                "kind": "trajectory",
                "sentence": sentence,
                "expected_current_nodes": {
                    source,
                    target,
                },
                "expected_next_nodes": {
                    source,
                    target,
                },
                "expected_edges": {
                    (
                        source,
                        relation,
                        target,
                    )
                },
            }
        )

    return cases


def load_model(
    device,
):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            MODEL_DIR.resolve()
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


def make_messages(
    kind: str,
    sentence: str,
) -> list[dict]:
    if kind == "relation":
        examples = "\n".join(
            f"User: {text}\nAssistant: {answer}"
            for text, answer
            in CLASSIFICATION_EXAMPLES
        )

        return [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\nFor relation tasks return exactly "
                    '{"relation":"RELATION_NAME"}.\n'
                    + examples
                ),
            },
            {
                "role": "user",
                "content": sentence,
            },
        ]

    if kind in {
        "single",
        "multi",
    }:
        examples = "\n".join(
            f"User: {text}\nAssistant: {answer}"
            for text, answer
            in GRAPH_EXAMPLES
        )

        return [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\nFor graph tasks return exactly "
                    '{"nodes":[...],"edges":[{"source":"...",'
                    '"relation":"...","target":"..."}]}.\n'
                    + examples
                ),
            },
            {
                "role": "user",
                "content": sentence,
            },
        ]

    if kind == "trajectory":
        example_text, example_json = (
            TRAJECTORY_EXAMPLE
        )
        return [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\nFor trajectory tasks return exactly "
                    '{"current_nodes":[...],'
                    '"next_nodes":[...],'
                    '"edges_added":[{"source":"...",'
                    '"relation":"...","target":"..."}]}.\n'
                    + "Example:\n"
                    + example_text
                    + "\nAssistant: "
                    + example_json
                ),
            },
            {
                "role": "user",
                "content": sentence,
            },
        ]

    raise ValueError(
        f"Unknown task kind: {kind}"
    )


def generate(
    tokenizer,
    model,
    device,
    messages,
    max_new_tokens,
) -> str:
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
        prompt = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.02,
        )

    generated = output[
        0
    ][
        inputs["input_ids"].shape[1] :
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--multi",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--trajectory",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
    )
    args = parser.parse_args()

    print(
        "=== V204B COMPREHENSIVE SMOLLM2 TEACHER QUALIFICATION ===",
        flush=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
        raise FileNotFoundError(DB_PATH.resolve())
    if not MODEL_DIR.exists():
        raise FileNotFoundError(MODEL_DIR.resolve())

    cases = build_cases(
        single=args.single,
        multi=args.multi,
        trajectory=args.trajectory,
    )

    expected_total = (
        args.single
        + args.single
        + args.multi
        + args.trajectory
    )

    if len(cases) != expected_total:
        raise RuntimeError(
            f"Case construction failed: "
            f"expected {expected_total}, got {len(cases)}"
        )

    counts = Counter(
        case["kind"]
        for case in cases
    )

    print(
        "cases:",
        len(cases),
        dict(counts),
        flush=True,
    )

    tokenizer, model = load_model(
        device
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    relation_correct = 0
    relation_count = 0

    graph_parse = 0
    graph_cases = 0

    node_tp = node_fp = node_fn = 0
    edge_tp = edge_fp = edge_fn = 0

    trajectory_parse = 0
    trajectory_cases = 0
    trajectory_edge_hit = 0

    raw_records = []

    started = time.perf_counter()

    with RESPONSES_PATH.open(
        "w",
        encoding="utf-8",
    ) as response_file:
        for index, case in enumerate(
            cases,
            1,
        ):
            messages = make_messages(
                case["kind"],
                case["sentence"],
            )

            raw = generate(
                tokenizer,
                model,
                device,
                messages,
                args.max_new_tokens,
            )

            record = {
                "id": case["id"],
                "kind": case["kind"],
                "sentence": case["sentence"],
                "raw_output": raw,
            }

            if case["kind"] == "relation":
                relation_count += 1

                predicted = parse_relation(
                    raw
                )

                correct = (
                    predicted
                    == case[
                        "expected_relation"
                    ]
                )

                relation_correct += int(
                    correct
                )

                record.update(
                    {
                        "expected_relation": case[
                            "expected_relation"
                        ],
                        "predicted_relation": predicted,
                        "correct": correct,
                    }
                )

            elif case["kind"] in {
                "single",
                "multi",
            }:
                graph_cases += 1

                (
                    predicted_nodes,
                    predicted_edges,
                    parse_error,
                ) = parse_graph(
                    raw
                )

                expected_nodes = set(
                    case[
                        "expected_nodes"
                    ]
                )
                expected_edges = set(
                    case[
                        "expected_edges"
                    ]
                )

                if parse_error is None:
                    graph_parse += 1

                tp = len(
                    predicted_nodes
                    & expected_nodes
                )
                fp = len(
                    predicted_nodes
                    - expected_nodes
                )
                fn = len(
                    expected_nodes
                    - predicted_nodes
                )

                node_tp += tp
                node_fp += fp
                node_fn += fn

                etp = len(
                    predicted_edges
                    & expected_edges
                )
                efp = len(
                    predicted_edges
                    - expected_edges
                )
                efn = len(
                    expected_edges
                    - predicted_edges
                )

                edge_tp += etp
                edge_fp += efp
                edge_fn += efn

                record.update(
                    {
                        "expected_nodes": sorted(
                            expected_nodes
                        ),
                        "expected_edges": sorted(
                            expected_edges
                        ),
                        "predicted_nodes": sorted(
                            predicted_nodes
                        ),
                        "predicted_edges": sorted(
                            predicted_edges
                        ),
                        "parse_error": parse_error,
                        "node_tp": tp,
                        "node_fp": fp,
                        "node_fn": fn,
                        "edge_tp": etp,
                        "edge_fp": efp,
                        "edge_fn": efn,
                    }
                )

            else:
                trajectory_cases += 1

                (
                    current_nodes,
                    next_nodes,
                    edges,
                    parse_error,
                ) = parse_trajectory(
                    raw
                )

                expected_current = set(
                    case[
                        "expected_current_nodes"
                    ]
                )
                expected_next = set(
                    case[
                        "expected_next_nodes"
                    ]
                )
                expected_edges = set(
                    case[
                        "expected_edges"
                    ]
                )

                if parse_error is None:
                    trajectory_parse += 1

                edge_hit = bool(
                    edges
                    & expected_edges
                )
                trajectory_edge_hit += int(
                    edge_hit
                )

                record.update(
                    {
                        "expected_current_nodes": sorted(
                            expected_current
                        ),
                        "expected_next_nodes": sorted(
                            expected_next
                        ),
                        "expected_edges": sorted(
                            expected_edges
                        ),
                        "predicted_current_nodes": sorted(
                            current_nodes
                        ),
                        "predicted_next_nodes": sorted(
                            next_nodes
                        ),
                        "predicted_edges": sorted(
                            edges
                        ),
                        "parse_error": parse_error,
                        "edge_hit": edge_hit,
                    }
                )

            response_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            response_file.flush()

            if (
                index <= 5
                or index % 10 == 0
                or index == len(cases)
            ):
                print(
                    f"CASE {index:3d}/{len(cases):3d} "
                    f"kind={case['kind']:10s} "
                    f"raw_chars={len(raw):4d}",
                    flush=True,
                )

            raw_records.append(
                record
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

    report = {
        "experiment": (
            "V204B comprehensive SmolLM2 teacher qualification"
        ),
        "model_dir": str(
            MODEL_DIR.resolve()
        ),
        "device": str(device),
        "case_counts": {
            "relation": relation_count,
            "single": args.single,
            "multi": args.multi,
            "trajectory": trajectory_cases,
            "total": len(cases),
        },
        "relation": {
            "accuracy": (
                relation_correct
                / max(
                    1,
                    relation_count,
                )
            )
        },
        "graph": {
            "parse_success": (
                graph_parse
                / max(
                    1,
                    graph_cases,
                )
            ),
            "node_precision": node_precision,
            "node_recall": node_recall,
            "node_f1": node_f1,
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            "hallucinated_edge_rate": (
                edge_fp
                / max(
                    1,
                    edge_tp + edge_fp,
                )
            ),
            "missed_edge_rate": (
                edge_fn
                / max(
                    1,
                    edge_tp + edge_fn,
                )
            ),
        },
        "trajectory": {
            "parse_success": (
                trajectory_parse
                / max(
                    1,
                    trajectory_cases,
                )
            ),
            "edge_hit_rate": (
                trajectory_edge_hit
                / max(
                    1,
                    trajectory_cases,
                )
            ),
        },
        "elapsed_seconds": (
            time.perf_counter()
            - started
        ),
        "responses_path": str(
            RESPONSES_PATH.resolve()
        ),
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
        "=== V204B SUMMARY ==="
    )
    print(
        "relation_accuracy:",
        f"{report['relation']['accuracy']:.4f}",
    )
    print(
        "graph_parse_success:",
        f"{report['graph']['parse_success']:.4f}",
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
        "hallucinated_edge_rate:",
        f"{report['graph']['hallucinated_edge_rate']:.4f}",
    )
    print(
        "missed_edge_rate:",
        f"{report['graph']['missed_edge_rate']:.4f}",
    )
    print(
        "trajectory_parse_success:",
        f"{report['trajectory']['parse_success']:.4f}",
    )
    print(
        "trajectory_edge_hit_rate:",
        f"{report['trajectory']['edge_hit_rate']:.4f}",
    )
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
