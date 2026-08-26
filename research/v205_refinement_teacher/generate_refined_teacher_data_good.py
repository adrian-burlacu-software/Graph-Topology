from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sqlite3
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Direct execution from research/:
#   python .\\v205_refinement_teacher\\generate_refined_teacher_data_good.py
_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


from dataclasses import dataclass, asdict
from typing import Any


ACTIONS = (
    "NOOP",
    "REUSE",
    "CREATE",
    "BRANCH",
    "INHIBIT",
    "BIND",
    "COMMIT",
)

ACTION_TO_ID = {
    action: i
    for i, action in enumerate(ACTIONS)
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    action: str
    source: str | None = None
    target: str | None = None
    relation: str | None = None


@dataclass
class Node:
    concept: str
    activation: float
    role: int
    persistent: bool = False


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    activation: float
    persistent: bool = False


@dataclass
class State:
    nodes: list[Node]
    edges: list[Edge]

    def clone(self) -> "State":
        return State(
            nodes=[
                Node(
                    concept=n.concept,
                    activation=n.activation,
                    role=n.role,
                    persistent=n.persistent,
                )
                for n in self.nodes
            ],
            edges=[
                Edge(
                    source=e.source,
                    relation=e.relation,
                    target=e.target,
                    activation=e.activation,
                    persistent=e.persistent,
                )
                for e in self.edges
            ],
        )

    def node(self, concept: str) -> Node | None:
        for node in self.nodes:
            if node.concept == concept:
                return node
        return None

    def add_node(
        self,
        concept: str,
        activation: float,
        role: int,
    ) -> None:
        existing = self.node(concept)
        if existing is not None:
            existing.activation = max(
                existing.activation,
                activation,
            )
            return

        self.nodes.append(
            Node(
                concept=concept,
                activation=activation,
                role=role,
            )
        )

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        activation: float = 1.0,
    ) -> None:
        for edge in self.edges:
            if (
                edge.source == source
                and edge.relation == relation
                and edge.target == target
            ):
                edge.activation = max(
                    edge.activation,
                    activation,
                )
                return

        self.edges.append(
            Edge(
                source=source,
                relation=relation,
                target=target,
                activation=activation,
            )
        )

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
        active_only: bool = True,
    ) -> bool:
        return any(
            edge.source == source
            and edge.relation == relation
            and edge.target == target
            and (
                not active_only
                or edge.activation > 0.5
            )
            for edge in self.edges
        )

    def signature(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "concept": n.concept,
                    "activation": round(
                        n.activation,
                        4,
                    ),
                    "role": n.role,
                    "persistent": n.persistent,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "relation": e.relation,
                    "target": e.target,
                    "activation": round(
                        e.activation,
                        4,
                    ),
                    "persistent": e.persistent,
                }
                for e in self.edges
            ],
        }


def candidate_dict(
    candidate: Candidate | None,
) -> dict | None:
    return (
        asdict(candidate)
        if candidate is not None
        else None
    )



ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LLM = ROOT / "llm"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_DIR = LLM / "SmolLM2-1.7B-Instruct"

RAW_PATH = (
    RESULTS
    / "v205r_teacher_refinement.jsonl"
)
ACCEPTED_PATH = (
    RESULTS
    / "v205r_teacher_dataset.jsonl"
)
SUMMARY_PATH = (
    RESULTS
    / "v205r_teacher_refinement_summary.json"
)

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
RELATION_SET = set(RELATIONS)


class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(
            str(path)
        )
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def source_edges(
        self,
        source: str,
        limit: int = 8,
    ) -> list[tuple[str, str, str, float]]:
        rows = self.conn.execute(
            """
            SELECT start, relation, end, weight
            FROM edge
            WHERE start = ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (source, limit),
        ).fetchall()

        return [
            (
                row["start"],
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
            for row in rows
            if row["relation"] in RELATION_SET
        ]


def build_scenario(
    db: ConceptNet,
    seed: str,
    neighbors: int = 6,
) -> tuple[State, dict]:
    edges = db.source_edges(
        seed,
        neighbors,
    )
    if not edges:
        raise RuntimeError(
            f"No valid semantic neighborhood for {seed!r}"
        )

    # Choose the strongest edge as the explicit goal.
    source, relation, target, _weight = edges[0]

    state = State(
        nodes=[],
        edges=[],
    )

    state.add_node(
        source,
        activation=1.0,
        role=1,
    )

    state.add_node(
        target,
        activation=0.0,
        role=2,
    )

    for edge_source, edge_relation, edge_target, _ in edges[1:]:
        if edge_target == target:
            continue

        state.add_node(
            edge_target,
            activation=0.35,
            role=3,
        )

        state.add_edge(
            edge_source,
            edge_relation,
            edge_target,
            activation=0.15,
        )

    # The goal is graph-derived and explicit.
    goal = {
        "source": source,
        "relation": relation,
        "target": target,
        "requirements": [
            f"activate node {target}",
            (
                f"activate edge "
                f"{source} --{relation}--> {target}"
            ),
        ],
    }

    return state, goal


def make_candidates(
    db: ConceptNet,
    state: State,
    goal: dict | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        action: str,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> None:
        candidates.append(
            Candidate(
                candidate_id=-1,
                action=action,
                source=source,
                target=target,
                relation=relation,
            )
        )

    add("NOOP")

    # REUSE: inactive nodes already present in working memory.
    for node in state.nodes:
        if node.activation < 0.5:
            add(
                "REUSE",
                target=node.concept,
            )

    # BIND: first add the actual goal edge, even if that edge is not already
    # represented in working memory. This is essential: otherwise the teacher
    # can be given a goal that is literally impossible from the candidate menu.
    if goal is not None:
        goal_source = goal.get("source")
        goal_target = goal.get("target")
        goal_relation = goal.get("relation")

        if (
            isinstance(goal_source, str)
            and isinstance(goal_target, str)
            and isinstance(goal_relation, str)
            and db.conn.execute(
                """
                SELECT 1
                FROM edge
                WHERE start = ?
                  AND relation = ?
                  AND end = ?
                LIMIT 1
                """,
                (
                    goal_source,
                    goal_relation,
                    goal_target,
                ),
            ).fetchone()
            is not None
            and state.node(goal_source) is not None
            and state.node(goal_target) is not None
            and not state.has_edge(
                goal_source,
                goal_relation,
                goal_target,
                active_only=True,
            )
        ):
            add(
                "BIND",
                source=goal_source,
                target=goal_target,
                relation=goal_relation,
            )

    # Other graph-valid BINDs from currently represented semantic edges.
    for edge in state.edges:
        if not state.has_edge(
            edge.source,
            edge.relation,
            edge.target,
            active_only=True,
        ):
            add(
                "BIND",
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
            )

    # INHIBIT: active non-source nodes.
    for node in state.nodes:
        if (
            node.activation > 0.5
            and node.role != 1
        ):
            add(
                "INHIBIT",
                target=node.concept,
            )

    # BRANCH: use real relations from the current semantic neighborhood.
    source = state.nodes[0].concept
    for edge in state.edges[:3]:
        add(
            "BRANCH",
            source=source,
            relation=edge.relation,
        )

    add("CREATE")

    if (
        any(
            node.activation > 0.5
            for node in state.nodes
        )
        or any(
            edge.activation > 0.5
            for edge in state.edges
        )
    ):
        add("COMMIT")

    unique: list[Candidate] = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate.action,
            candidate.source,
            candidate.target,
            candidate.relation,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return [
        Candidate(
            candidate_id=index,
            action=candidate.action,
            source=candidate.source,
            target=candidate.target,
            relation=candidate.relation,
        )
        for index, candidate in enumerate(unique)
    ]


def validate_candidate(
    db: ConceptNet,
    state: State,
    candidate: Candidate,
) -> tuple[bool, str]:
    if candidate.action not in ACTION_TO_ID:
        return False, "unknown_action"

    if candidate.action == "NOOP":
        return True, "ok"

    if candidate.action == "CREATE":
        return True, "ok"

    if candidate.action == "COMMIT":
        if (
            any(
                node.activation > 0.5
                for node in state.nodes
            )
            or any(
                edge.activation > 0.5
                for edge in state.edges
            )
        ):
            return True, "ok"
        return False, "nothing_to_commit"

    if candidate.action == "REUSE":
        if candidate.target is None:
            return False, "missing_target"
        if state.node(
            candidate.target
        ) is None:
            return False, "unknown_target"
        return True, "ok"

    if candidate.action == "INHIBIT":
        if candidate.target is None:
            return False, "missing_target"
        node = state.node(
            candidate.target
        )
        if node is None:
            return False, "unknown_target"
        if node.activation <= 0.5:
            return False, "not_active"
        return True, "ok"

    if candidate.action == "BRANCH":
        if (
            candidate.source is None
            or candidate.relation is None
        ):
            return False, "missing_branch_args"
        if state.node(
            candidate.source
        ) is None:
            return False, "unknown_source"
        if candidate.relation not in RELATION_SET:
            return False, "unknown_relation"
        return True, "ok"

    if candidate.action == "BIND":
        if (
            candidate.source is None
            or candidate.target is None
            or candidate.relation is None
        ):
            return False, "missing_bind_args"

        if (
            state.node(
                candidate.source
            ) is None
            or state.node(
                candidate.target
            ) is None
        ):
            return False, "unknown_node"

        if not db.conn.execute(
            """
            SELECT 1
            FROM edge
            WHERE start = ?
              AND relation = ?
              AND end = ?
            LIMIT 1
            """,
            (
                candidate.source,
                candidate.relation,
                candidate.target,
            ),
        ).fetchone():
            return False, "not_in_long_term_memory"

        if state.has_edge(
            candidate.source,
            candidate.relation,
            candidate.target,
            active_only=True,
        ):
            return False, "already_active"

        return True, "ok"

    return False, "unhandled"


def apply_candidate(
    state: State,
    candidate: Candidate,
) -> State:
    result = state.clone()

    if candidate.action == "NOOP":
        return result

    if candidate.action == "CREATE":
        result.add_node(
            f"created_{len(result.nodes)}",
            activation=0.85,
            role=6,
        )
        return result

    if candidate.action == "REUSE":
        if candidate.target is not None:
            node = result.node(
                candidate.target
            )
            if node is not None:
                node.activation = 1.0
        return result

    if candidate.action == "INHIBIT":
        if candidate.target is not None:
            node = result.node(
                candidate.target
            )
            if node is not None:
                node.activation *= 0.05
        return result

    if candidate.action == "BRANCH":
        if (
            candidate.source is not None
            and candidate.relation is not None
        ):
            branch = (
                f"{candidate.source}"
                f"#branch{len(result.nodes)}"
            )
            result.add_node(
                branch,
                activation=0.8,
                role=7,
            )
            result.add_edge(
                candidate.source,
                candidate.relation,
                branch,
                activation=0.8,
            )
        return result

    if candidate.action == "BIND":
        if (
            candidate.source is not None
            and candidate.target is not None
            and candidate.relation is not None
        ):
            source_node = result.node(
                candidate.source
            )
            target_node = result.node(
                candidate.target
            )

            if source_node is not None:
                source_node.activation = max(
                    source_node.activation,
                    1.0,
                )

            if target_node is not None:
                target_node.activation = max(
                    target_node.activation,
                    1.0,
                )

            result.add_edge(
                candidate.source,
                candidate.relation,
                candidate.target,
                activation=1.0,
            )
        return result

    if candidate.action == "COMMIT":
        for node in result.nodes:
            if node.activation > 0.5:
                node.persistent = True

        for edge in result.edges:
            if edge.activation > 0.5:
                edge.persistent = True

        return result

    raise ValueError(
        candidate.action
    )


def goal_status(
    state: State,
    goal: dict,
) -> dict:
    source = goal["source"]
    relation = goal["relation"]
    target = goal["target"]

    target_node = state.node(
        target
    )

    target_active = (
        target_node is not None
        and target_node.activation > 0.5
    )

    edge_active = state.has_edge(
        source,
        relation,
        target,
        active_only=True,
    )

    return {
        "target_active": target_active,
        "edge_active": edge_active,
        "goal_reached": (
            target_active
            and edge_active
        ),
    }


def goal_distance(
    status: dict,
) -> int:
    return int(
        not status["target_active"]
    ) + int(
        not status["edge_active"]
    )


def parse_teacher(
    raw: str,
) -> tuple[int | None, float | None]:
    raw = raw.strip()

    # Preferred format: JSON.
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        value = None

        if (
            start >= 0
            and end > start
        ):
            try:
                value = json.loads(
                    raw[start : end + 1]
                )
            except json.JSONDecodeError:
                value = None

    if isinstance(value, dict):
        try:
            candidate_id = int(
                value.get("candidate_id")
            )
        except (
            TypeError,
            ValueError,
        ):
            candidate_id = None

        try:
            confidence = float(
                value.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = None

        if confidence is not None:
            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )

        if candidate_id is not None:
            return candidate_id, confidence

    # Fallback: SmolLM2 often emits:
    #   3: BIND | source=witchlike, target=witch, relation=RelatedTo.
    # Accept that form rather than rejecting an otherwise usable answer.
    patterns = (
        r"\b(?:candidate[_ ]?id|choice|option)\s*[:=]\s*(\d+)\b",
        r"^\s*(\d+)\s*:\s*[A-Z]+",
        r"\b(?:candidate|choice|option)\s+(\d+)\b",
        r"\b(\d+)\s*:\s*(?:NOOP|REUSE|CREATE|BRANCH|INHIBIT|BIND|COMMIT)\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            raw,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match:
            try:
                return int(
                    match.group(1)
                ), None
            except (
                TypeError,
                ValueError,
            ):
                pass

    return None, None


class Teacher:
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
    ):
        dtype = (
            torch.float16
            if device.type == "cuda"
            else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )

        self.model.to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.device = device

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> str:
        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            prompt = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            prompt = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in messages
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.02,
            )

        generated = output[
            0
        ][
            inputs["input_ids"].shape[1] :
        ]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


def candidate_text(
    candidates: list[Candidate],
) -> str:
    lines = []

    for candidate in candidates:
        details = []
        if candidate.source:
            details.append(
                f"source={candidate.source}"
            )
        if candidate.target:
            details.append(
                f"target={candidate.target}"
            )
        if candidate.relation:
            details.append(
                f"relation={candidate.relation}"
            )

        suffix = (
            " | "
            + ", ".join(details)
            if details
            else ""
        )

        lines.append(
            f"{candidate.candidate_id}: "
            f"{candidate.action}{suffix}"
        )

    return "\n".join(lines)


def first_turn_prompt(
    state: State,
    goal: dict,
    candidates: list[Candidate],
) -> list[dict]:
    state_text = json.dumps(
        state.signature(),
        indent=2,
        ensure_ascii=False,
    )

    goal_text = json.dumps(
        goal,
        indent=2,
        ensure_ascii=False,
    )

    candidate_text_value = candidate_text(
        candidates
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a cognitive planning teacher. "
                "Choose one action from the supplied valid candidates. "
                "The goal is explicit. "
                "Return ONLY "
                '{"candidate_id":0,"confidence":0.0}.'
            ),
        },
        {
            "role": "user",
            "content": (
                "CURRENT WORKING MEMORY:\n"
                f"{state_text}\n\n"
                "GOAL:\n"
                f"{goal_text}\n\n"
                "VALID CANDIDATES:\n"
                f"{candidate_text_value}\n\n"
                "Choose the single action that most directly reduces "
                "the remaining goal distance."
            ),
        },
    ]


def refinement_prompt(
    state: State,
    goal: dict,
    candidates: list[Candidate],
    first_candidate: Candidate | None,
    first_result: State,
    first_status: dict,
    first_distance: int,
) -> list[dict]:
    candidate_lines = "\n".join(
        (
            f"{candidate.candidate_id}: "
            f"{candidate.action}"
            + (
                f" | source={candidate.source}"
                if candidate.source
                else ""
            )
            + (
                f" | target={candidate.target}"
                if candidate.target
                else ""
            )
            + (
                f" | relation={candidate.relation}"
                if candidate.relation
                else ""
            )
        )
        for candidate in candidates
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a cognitive planning teacher reviewing your own "
                "previous decision. Your first decision may be wrong. "
                "Use the validator feedback and choose ONE candidate from "
                "the supplied VALID candidates. "
                "Return ONLY JSON like "
                '{"candidate_id":0,"confidence":0.0} '
                "or a single line such as "
                "0: BIND."
            ),
        },
        {
            "role": "user",
            "content": (
                "ORIGINAL STATE:\n"
                f"{json.dumps(state.signature(), indent=2, ensure_ascii=False)}\n\n"
                "GOAL:\n"
                f"{json.dumps(goal, indent=2, ensure_ascii=False)}\n\n"
                "YOUR FIRST CHOICE:\n"
                f"{json.dumps(candidate_dict(first_candidate), ensure_ascii=False)}\n\n"
                "STATE AFTER FIRST CHOICE:\n"
                f"{json.dumps(first_result.signature(), indent=2, ensure_ascii=False)}\n\n"
                "VALIDATOR FEEDBACK:\n"
                f"goal_reached={first_status['goal_reached']}\n"
                f"target_active={first_status['target_active']}\n"
                f"edge_active={first_status['edge_active']}\n"
                f"goal_distance={first_distance}\n\n"
                "VALID CANDIDATES FOR REVISION:\n"
                f"{candidate_lines}\n\n"
                "Choose the candidate that most directly reduces the remaining "
                "goal distance. Do not choose NOOP unless the goal is already "
                "reached."
            ),
        },
    ]


def load_cached_ids() -> set[str]:
    if not RAW_PATH.exists():
        return set()

    done = set()

    with RAW_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            try:
                row = json.loads(
                    line
                )
            except json.JSONDecodeError:
                continue

            if row.get(
                "case_id"
            ):
                done.add(
                    row["case_id"]
                )

    return done


def build_scenarios(
    db: ConceptNet,
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    rows = db.conn.execute(
        """
        SELECT start
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
        GROUP BY start
        HAVING COUNT(*) >= 2
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (
            max(
                count * 3,
                1000,
            ),
        ),
    ).fetchall()

    seeds = [
        row["start"]
        for row in rows
        if row["start"]
    ]

    random.Random(
        seed
    ).shuffle(
        seeds
    )

    seeds = list(
        dict.fromkeys(
            seeds
        )
    )

    if len(seeds) < count:
        raise RuntimeError(
            f"Could only create {len(seeds)} scenarios; requested {count}."
        )

    return [
        (
            f"scenario_{i:05d}",
            seed,
        )
        for i, seed in enumerate(
            seeds[:count],
            1,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2051,
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
        "=== V205R TWO-TURN TEACHER REFINEMENT ===",
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
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            MODEL_DIR.resolve()
        )

    db = ConceptNet(
        DB_PATH
    )
    teacher = Teacher(
        MODEL_DIR,
        device,
    )

    existing = load_cached_ids()

    scenarios = build_scenarios(
        db,
        args.cases,
        args.seed,
    )

    print(
        "requested_cases:",
        args.cases,
        flush=True,
    )
    print(
        "already_cached:",
        len(existing),
        flush=True,
    )

    stats = Counter()
    started = time.perf_counter()

    try:
        with RAW_PATH.open(
            "a",
            encoding="utf-8",
        ) as raw_file, ACCEPTED_PATH.open(
            "a",
            encoding="utf-8",
        ) as accepted_file:

            for processed, (
                case_id,
                seed,
            ) in enumerate(
                scenarios,
                1,
            ):
                if case_id in existing:
                    continue

                state, goal = build_scenario(
                    db,
                    seed,
                    neighbors=args.max_neighbors,
                )

                candidates = make_candidates(
                    db,
                    state,
                    goal,
                )

                if not candidates:
                    stats[
                        "no_candidates"
                    ] += 1
                    continue

                before_status = goal_status(
                    state,
                    goal,
                )
                before_distance = goal_distance(
                    before_status
                )

                # -------------------------------------------------------
                # TURN 1
                # -------------------------------------------------------
                first_raw = teacher.generate(
                    first_turn_prompt(
                        state,
                        goal,
                        candidates,
                    ),
                    args.max_new_tokens,
                )

                first_id, first_confidence = (
                    parse_teacher(
                        first_raw
                    )
                )

                first_candidate = next(
                    (
                        candidate
                        for candidate
                        in candidates
                        if candidate.candidate_id
                        == first_id
                    ),
                    None,
                )

                first_valid = False
                first_validation_reason = (
                    "no_candidate"
                )
                first_state = state.clone()

                if first_candidate is not None:
                    (
                        first_valid,
                        first_validation_reason,
                    ) = validate_candidate(
                        db,
                        state,
                        first_candidate,
                    )

                    if first_valid:
                        first_state = apply_candidate(
                            state,
                            first_candidate,
                        )

                first_status = goal_status(
                    first_state,
                    goal,
                )
                first_distance = goal_distance(
                    first_status
                )

                # -------------------------------------------------------
                # TURN 2 / REFINEMENT
                # -------------------------------------------------------
                refinement_needed = not (
                    first_status[
                        "goal_reached"
                    ]
                )

                second_raw = None
                second_id = None
                second_confidence = None
                second_candidate = None
                second_valid = False
                second_validation_reason = (
                    "not_needed"
                )
                final_state = first_state.clone()

                if refinement_needed:
                    stats[
                        "refinement_needed"
                    ] += 1

                    second_raw = teacher.generate(
                        refinement_prompt(
                            state,
                            goal,
                            candidates,
                            first_candidate,
                            first_state,
                            first_status,
                            first_distance,
                        ),
                        args.max_new_tokens,
                    )

                    (
                        second_id,
                        second_confidence,
                    ) = parse_teacher(
                        second_raw
                    )

                    second_candidate = next(
                        (
                            candidate
                            for candidate
                            in candidates
                            if candidate.candidate_id
                            == second_id
                        ),
                        None,
                    )

                    if (
                        second_candidate
                        is not None
                    ):
                        (
                            second_valid,
                            second_validation_reason,
                        ) = validate_candidate(
                            db,
                            first_state,
                            second_candidate,
                        )

                        if second_valid:
                            final_state = apply_candidate(
                                first_state,
                                second_candidate,
                            )

                final_status = goal_status(
                    final_state,
                    goal,
                )

                first_success = (
                    first_status[
                        "goal_reached"
                    ]
                )
                final_success = (
                    final_status[
                        "goal_reached"
                    ]
                )

                correction = (
                    not first_success
                    and final_success
                )

                if first_success:
                    stats[
                        "first_success"
                    ] += 1
                if final_success:
                    stats[
                        "final_success"
                    ] += 1
                if correction:
                    stats[
                        "corrected"
                    ] += 1
                if (
                    first_candidate is not None
                ):
                    stats[
                        "first_parsed"
                    ] += 1
                if (
                    second_candidate is not None
                ):
                    stats[
                        "second_parsed"
                    ] += 1
                if first_valid:
                    stats[
                        "first_valid"
                    ] += 1
                if second_valid:
                    stats[
                        "second_valid"
                    ] += 1

                record = {
                    "case_id": case_id,
                    "seed": seed,
                    "goal": goal,
                    "initial_state": state.signature(),
                    "candidates": [
                        candidate_dict(
                            candidate
                        )
                        for candidate in candidates
                    ],
                    "turn1": {
                        "raw_output": first_raw,
                        "parsed_candidate_id": first_id,
                        "confidence": first_confidence,
                        "candidate": candidate_dict(
                            first_candidate
                        ),
                        "valid": first_valid,
                        "validation_reason": (
                            first_validation_reason
                        ),
                        "state_after": first_state.signature(),
                        "goal_status": first_status,
                        "goal_distance_before": before_distance,
                        "goal_distance_after": first_distance,
                    },
                    "turn2": {
                        "refinement_needed": refinement_needed,
                        "raw_output": second_raw,
                        "parsed_candidate_id": second_id,
                        "confidence": second_confidence,
                        "candidate": candidate_dict(
                            second_candidate
                        ),
                        "valid": second_valid,
                        "validation_reason": (
                            second_validation_reason
                        ),
                        "state_after": final_state.signature(),
                        "goal_status": final_status,
                        "goal_distance_after": goal_distance(
                            final_status
                        ),
                    },
                    "first_success": first_success,
                    "final_success": final_success,
                    "corrected": correction,
                }

                raw_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()

                if final_success:
                    accepted_record = {
                        "case_id": case_id,
                        "seed": seed,
                        "goal": goal,
                        "initial_state": state.signature(),
                        "candidates": [
                            candidate_dict(
                                candidate
                            )
                            for candidate
                            in candidates
                        ],
                        "teacher_turn1": {
                            "candidate": candidate_dict(
                                first_candidate
                            ),
                            "confidence": first_confidence,
                            "success": first_success,
                        },
                        "teacher_turn2": {
                            "candidate": candidate_dict(
                                second_candidate
                            ),
                            "confidence": second_confidence,
                            "used": refinement_needed,
                            "corrected": correction,
                        },
                        "final_action": candidate_dict(
                            second_candidate
                            if second_candidate is not None
                            else first_candidate
                        ),
                        "final_state": final_state.signature(),
                        "action_sequence": [
                            item
                            for item in [
                                candidate_dict(
                                    first_candidate
                                )
                                if first_success
                                else None,
                                candidate_dict(
                                    second_candidate
                                )
                                if correction
                                else None,
                            ]
                            if item is not None
                        ],
                    }

                    accepted_file.write(
                        json.dumps(
                            accepted_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted_file.flush()

                existing.add(
                    case_id
                )

                done = (
                    stats[
                        "first_success"
                    ]
                    + (
                        stats[
                            "refinement_needed"
                        ]
                        - (
                            stats[
                                "final_success"
                            ]
                            - stats[
                                "first_success"
                            ]
                        )
                    )
                )
                # Use actual newly-completed case count instead of the
                # approximate helper above for display.
                completed = (
                    stats["first_success"]
                    + stats["corrected"]
                    + (
                        max(
                            0,
                            stats["refinement_needed"]
                            - stats["corrected"],
                        )
                    )
                    + max(
                        0,
                        processed
                        - 1
                        - (
                            stats["first_success"]
                            + stats["refinement_needed"]
                        ),
                    )
                )
                # More reliable progress counter:
                stats[
                    "processed"
                ] += 1

                if (
                    stats["processed"] <= 5
                    or stats["processed"] % 25 == 0
                    or stats["processed"] == args.cases
                ):
                    print(
                        f"TEACHER "
                        f"{stats['processed']}/{args.cases} "
                        f"first={stats['first_success']} "
                        f"final={stats['final_success']} "
                        f"corrected={stats['corrected']}",
                        flush=True,
                    )

        processed = stats["processed"]

        summary = {
            "experiment": (
                "V205R two-turn teacher refinement"
            ),
            "requested_cases": args.cases,
            "processed_this_run": processed,
            "first_turn_success_rate": (
                stats["first_success"]
                / max(
                    1,
                    processed,
                )
            ),
            "final_success_rate": (
                stats["final_success"]
                / max(
                    1,
                    processed,
                )
            ),
            "correction_rate": (
                stats["corrected"]
                / max(
                    1,
                    processed,
                )
            ),
            "refinement_needed_rate": (
                stats["refinement_needed"]
                / max(
                    1,
                    processed,
                )
            ),
            "first_parse_rate": (
                stats["first_parsed"]
                / max(
                    1,
                    processed,
                )
            ),
            "second_parse_rate": (
                stats["second_parsed"]
                / max(
                    1,
                    stats["refinement_needed"],
                )
            ),
            "first_valid_rate": (
                stats["first_valid"]
                / max(
                    1,
                    processed,
                )
            ),
            "second_valid_rate": (
                stats["second_valid"]
                / max(
                    1,
                    stats["refinement_needed"],
                )
            ),
            "raw_path": str(
                RAW_PATH.resolve()
            ),
            "accepted_path": str(
                ACCEPTED_PATH.resolve()
            ),
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
        }

        SUMMARY_PATH.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "=== V205R COMPLETE ==="
        )
        print(
            "processed:",
            processed,
        )
        print(
            "first_turn_success:",
            summary[
                "first_turn_success_rate"
            ],
        )
        print(
            "final_success:",
            summary[
                "final_success_rate"
            ],
        )
        print(
            "correction_rate:",
            summary[
                "correction_rate"
            ],
        )
        print(
            "refinement_needed:",
            summary[
                "refinement_needed_rate"
            ],
        )
        print(
            "raw_cache:",
            RAW_PATH.resolve(),
        )
        print(
            "accepted_dataset:",
            ACCEPTED_PATH.resolve(),
        )
        print(
            "summary:",
            SUMMARY_PATH.resolve(),
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
