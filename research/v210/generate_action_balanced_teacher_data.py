from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Direct execution from research/:
#   python .\v210_action_balanced_teacher\generate_action_balanced_teacher_data.py
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LLM = ROOT / "llm"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_DIR = LLM / "SmolLM2-1.7B-Instruct"

RAW_PATH = RESULTS / "v210_action_balanced_teacher_refinement.jsonl"
ACCEPTED_PATH = RESULTS / "v210_action_balanced_teacher_dataset.jsonl"
SUMMARY_PATH = RESULTS / "v210_action_balanced_teacher_summary.json"

ACTIONS = (
    "NOOP",
    "REUSE",
    "CREATE",
    "BRANCH",
    "INHIBIT",
    "BIND",
    "COMMIT",
)
ACTION_TO_ID = {action: i for i, action in enumerate(ACTIONS)}

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
                Node(n.concept, n.activation, n.role, n.persistent)
                for n in self.nodes
            ],
            edges=[
                Edge(e.source, e.relation, e.target, e.activation, e.persistent)
                for e in self.edges
            ],
        )

    def node(self, concept: str) -> Node | None:
        return next((n for n in self.nodes if n.concept == concept), None)

    def add_node(self, concept: str, activation: float, role: int) -> None:
        existing = self.node(concept)
        if existing is not None:
            existing.activation = max(existing.activation, activation)
            return
        self.nodes.append(Node(concept, activation, role))

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
                edge.activation = max(edge.activation, activation)
                return
        self.edges.append(Edge(source, relation, target, activation))

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
            and (not active_only or edge.activation > 0.5)
            for edge in self.edges
        )

    def signature(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "concept": n.concept,
                    "activation": round(n.activation, 4),
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
                    "activation": round(e.activation, 4),
                    "persistent": e.persistent,
                }
                for e in self.edges
            ],
        }


def candidate_dict(candidate: Candidate | None) -> dict | None:
    return asdict(candidate) if candidate is not None else None


class ConceptNet:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
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
              AND relation IN (
                'IsA', 'CapableOf', 'HasProperty', 'UsedFor', 'HasA',
                'PartOf', 'RelatedTo', 'SimilarTo', 'Antonym', 'Causes',
                'AtLocation'
              )
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
        ]

    def edge_exists(self, source: str, relation: str, target: str) -> bool:
        return (
            self.conn.execute(
                """
                SELECT 1
                FROM edge
                WHERE start = ?
                  AND relation = ?
                  AND end = ?
                LIMIT 1
                """,
                (source, relation, target),
            ).fetchone()
            is not None
        )


def build_base_state(
    db: ConceptNet,
    seed: str,
    rng: random.Random,
    neighbors: int,
) -> tuple[State, tuple[str, str, str], list[tuple[str, str, str]]]:
    edges = db.source_edges(seed, max(2, neighbors))
    if not edges:
        raise RuntimeError(f"No valid semantic neighborhood for {seed!r}")

    source, relation, target, _ = edges[0]

    state = State(nodes=[], edges=[])

    # Primary source node.
    state.add_node(source, activation=1.0, role=1)

    # Goal target starts inactive.
    state.add_node(target, activation=0.0, role=2)

    distractor_edges: list[tuple[str, str, str]] = []
    for edge_source, edge_relation, edge_target, _ in edges[1:]:
        if edge_target == target or edge_target == source:
            continue

        # Keep at least one active distractor in working memory so that
        # INHIBIT remains a represented candidate class across the dataset.
        distractor_activation = 0.8 if not distractor_edges else 0.35
        state.add_node(
            edge_target,
            activation=distractor_activation,
            role=3,
        )
        state.add_edge(
            edge_source,
            edge_relation,
            edge_target,
            activation=0.15,
        )
        distractor_edges.append(
            (edge_source, edge_relation, edge_target)
        )

    rng.shuffle(distractor_edges)
    return state, (source, relation, target), distractor_edges


def build_action_scenario(
    db: ConceptNet,
    seed: str,
    requested_action: str,
    rng: random.Random,
    neighbors: int,
) -> tuple[State, dict]:
    state, primary_edge, distractors = build_base_state(
        db,
        seed,
        rng,
        neighbors,
    )
    source, relation, target = primary_edge

    # The important property here is that every action gets a task for which
    # that action is the direct, validator-checked solution.
    if requested_action == "NOOP":
        # Goal is already satisfied. The only accepted teacher action is NOOP.
        target_node = state.node(target)
        assert target_node is not None
        target_node.activation = 1.0
        state.add_edge(source, relation, target, activation=1.0)
        goal = {
            "objective": "leave_satisfied_state_unchanged",
            "description": (
                "The goal condition is already satisfied. Do not change the "
                "working-memory state."
            ),
            "action_family": "NOOP",
            "source": source,
            "relation": relation,
            "target": target,
        }
        return state, goal

    if requested_action == "REUSE":
        # Existing inactive concept must become active; no new semantic edge
        # is required.
        goal = {
            "objective": "activate_existing_concept",
            "description": (
                f"Activate the existing working-memory concept '{target}'. "
                "Do not require a new semantic edge."
            ),
            "action_family": "REUSE",
            "target": target,
        }
        return state, goal

    if requested_action == "BIND":
        goal = {
            "objective": "activate_semantic_edge",
            "description": (
                f"Activate the semantic relation {source} --{relation}--> "
                f"{target} and activate both participating concepts."
            ),
            "action_family": "BIND",
            "source": source,
            "relation": relation,
            "target": target,
        }
        return state, goal

    if requested_action == "INHIBIT":
        # Prefer a true distractor node that is currently active and not the
        # primary source.
        candidates = [
            edge_target
            for _, _, edge_target in distractors
            if edge_target != target
        ]
        if not candidates:
            # Fallback: the primary target can be made active only for this
            # task, because the goal is suppression rather than binding.
            candidates = [target]

        inhibit_target = rng.choice(candidates)
        node = state.node(inhibit_target)
        assert node is not None
        node.activation = 1.0
        node.role = 3

        goal = {
            "objective": "suppress_active_concept",
            "description": (
                f"Reduce the activation of the distracting concept "
                f"'{inhibit_target}' below the active threshold."
            ),
            "action_family": "INHIBIT",
            "target": inhibit_target,
        }
        return state, goal

    if requested_action == "BRANCH":
        # The branch action creates a fresh branch node from the source using
        # an existing real semantic relation.
        branch_relation = (
            distractors[0][1] if distractors else relation
        )
        goal = {
            "objective": "create_branch",
            "description": (
                f"Create a new working-memory branch from '{source}' using "
                f"the relation '{branch_relation}'."
            ),
            "action_family": "BRANCH",
            "source": source,
            "relation": branch_relation,
        }
        return state, goal

    if requested_action == "CREATE":
        goal = {
            "objective": "create_new_concept",
            "description": (
                "Create one new working-memory concept that does not already "
                "exist in the current graph."
            ),
            "action_family": "CREATE",
        }
        return state, goal

    if requested_action == "COMMIT":
        active_targets = [n.concept for n in state.nodes if n.activation > 0.5]
        if not active_targets:
            state.node(source).activation = 1.0  # defensive
            active_targets = [source]

        commit_target = rng.choice(active_targets)
        goal = {
            "objective": "make_knowledge_persistent",
            "description": (
                f"Make the active working-memory concept '{commit_target}' "
                "persistent."
            ),
            "action_family": "COMMIT",
            "target": commit_target,
        }
        return state, goal

    raise ValueError(f"Unknown requested action: {requested_action}")


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

    # Every task sees every action family whenever the state permits a valid
    # argument set. This avoids a classifier learning action priors from a
    # missing candidate menu.
    add("NOOP")

    for node in state.nodes:
        if node.activation < 0.5:
            add("REUSE", target=node.concept)

    add("CREATE")

    branch_source = next(
        (node.concept for node in state.nodes if node.role == 1),
        state.nodes[0].concept,
    )
    branch_relations = [
        edge.relation
        for edge in state.edges
        if edge.relation in RELATION_SET
    ]
    if not branch_relations:
        branch_relations = ["RelatedTo"]
    for branch_relation in dict.fromkeys(branch_relations[:3]):
        add(
            "BRANCH",
            source=branch_source,
            relation=branch_relation,
        )

    for node in state.nodes:
        if node.activation > 0.5 and node.role != 1:
            add("INHIBIT", target=node.concept)

    # Candidate BINDs from all represented inactive semantic edges.
    for edge in state.edges:
        if edge.relation in RELATION_SET and not state.has_edge(
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

    # Also inject the explicit BIND goal when that edge is in long-term memory.
    if goal is not None and goal.get("objective") == "activate_semantic_edge":
        goal_source = goal.get("source")
        goal_target = goal.get("target")
        goal_relation = goal.get("relation")
        if (
            isinstance(goal_source, str)
            and isinstance(goal_target, str)
            and isinstance(goal_relation, str)
            and db.edge_exists(goal_source, goal_relation, goal_target)
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

    active_nodes = [n for n in state.nodes if n.activation > 0.5]
    if active_nodes:
        add("COMMIT", target=active_nodes[0].concept)

    # Deduplicate while preserving ordering.
    unique: list[Candidate] = []
    seen: set[tuple] = set()
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
            candidate_id=i,
            action=candidate.action,
            source=candidate.source,
            target=candidate.target,
            relation=candidate.relation,
        )
        for i, candidate in enumerate(unique)
    ]


def validate_candidate(
    db: ConceptNet,
    state: State,
    candidate: Candidate,
) -> tuple[bool, str]:
    if candidate.action not in ACTION_TO_ID:
        return False, "unknown_action"

    if candidate.action in {"NOOP", "CREATE"}:
        return True, "ok"

    if candidate.action == "COMMIT":
        if any(n.activation > 0.5 for n in state.nodes) or any(
            e.activation > 0.5 for e in state.edges
        ):
            return True, "ok"
        return False, "nothing_to_commit"

    if candidate.action == "REUSE":
        if candidate.target is None:
            return False, "missing_target"
        node = state.node(candidate.target)
        if node is None:
            return False, "unknown_target"
        if node.activation >= 0.5:
            return False, "already_active"
        return True, "ok"

    if candidate.action == "INHIBIT":
        if candidate.target is None:
            return False, "missing_target"
        node = state.node(candidate.target)
        if node is None:
            return False, "unknown_target"
        if node.activation <= 0.5:
            return False, "not_active"
        return True, "ok"

    if candidate.action == "BRANCH":
        if candidate.source is None or candidate.relation is None:
            return False, "missing_branch_args"
        if state.node(candidate.source) is None:
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
            state.node(candidate.source) is None
            or state.node(candidate.target) is None
        ):
            return False, "unknown_node"
        if not db.edge_exists(
            candidate.source,
            candidate.relation,
            candidate.target,
        ):
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


def apply_candidate(state: State, candidate: Candidate) -> State:
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
        node = result.node(candidate.target) if candidate.target else None
        if node is not None:
            node.activation = 1.0
        return result

    if candidate.action == "INHIBIT":
        node = result.node(candidate.target) if candidate.target else None
        if node is not None:
            node.activation *= 0.05
        return result

    if candidate.action == "BRANCH":
        if candidate.source is not None and candidate.relation is not None:
            branch = f"{candidate.source}#branch{len(result.nodes)}"
            result.add_node(branch, activation=0.8, role=7)
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
            source_node = result.node(candidate.source)
            target_node = result.node(candidate.target)
            if source_node is not None:
                source_node.activation = max(source_node.activation, 1.0)
            if target_node is not None:
                target_node.activation = max(target_node.activation, 1.0)
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

    raise ValueError(candidate.action)


def goal_status(state: State, goal: dict) -> dict:
    objective = goal.get("objective")

    if objective == "leave_satisfied_state_unchanged":
        source = goal["source"]
        relation = goal["relation"]
        target = goal["target"]
        target_node = state.node(target)
        target_active = target_node is not None and target_node.activation > 0.5
        edge_active = state.has_edge(source, relation, target, active_only=True)
        return {
            "goal_reached": target_active and edge_active,
            "target_active": target_active,
            "edge_active": edge_active,
        }

    if objective == "activate_existing_concept":
        node = state.node(goal["target"])
        target_active = node is not None and node.activation > 0.5
        return {
            "goal_reached": target_active,
            "target_active": target_active,
        }

    if objective == "activate_semantic_edge":
        source = goal["source"]
        relation = goal["relation"]
        target = goal["target"]
        target_node = state.node(target)
        target_active = target_node is not None and target_node.activation > 0.5
        edge_active = state.has_edge(source, relation, target, active_only=True)
        return {
            "goal_reached": target_active and edge_active,
            "target_active": target_active,
            "edge_active": edge_active,
        }

    if objective == "suppress_active_concept":
        node = state.node(goal["target"])
        suppressed = node is None or node.activation <= 0.5
        return {
            "goal_reached": suppressed,
            "suppressed": suppressed,
        }

    if objective == "create_branch":
        source = goal["source"]
        relation = goal["relation"]
        exists = any(
            edge.source == source
            and edge.relation == relation
            and edge.activation > 0.5
            and "#branch" in edge.target
            for edge in state.edges
        )
        return {"goal_reached": exists, "branch_exists": exists}

    if objective == "create_new_concept":
        created = any(n.role == 6 for n in state.nodes)
        return {"goal_reached": created, "created": created}

    if objective == "make_knowledge_persistent":
        node = state.node(goal["target"])
        persistent = node is not None and node.persistent
        return {"goal_reached": persistent, "persistent": persistent}

    raise ValueError(f"Unknown goal objective: {objective}")


def goal_distance(status: dict) -> int:
    if "goal_reached" in status:
        return 0 if status["goal_reached"] else 1
    return 0


def parse_teacher(raw: str) -> tuple[int | None, float | None]:
    raw = raw.strip()

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        value = None
        if start >= 0 and end > start:
            try:
                value = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                value = None

    if isinstance(value, dict):
        try:
            candidate_id = int(value.get("candidate_id"))
        except (TypeError, ValueError):
            candidate_id = None

        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = None

        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))

        if candidate_id is not None:
            return candidate_id, confidence

    patterns = (
        r"\b(?:candidate[_ ]?id|choice|option)\s*[:=]\s*(\d+)\b",
        r"^\s*(\d+)\s*:\s*[A-Z]+",
        r"\b(?:candidate|choice|option)\s+(\d+)\b",
        r"\b(\d+)\s*:\s*(?:NOOP|REUSE|CREATE|BRANCH|INHIBIT|BIND|COMMIT)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1)), None
            except (TypeError, ValueError):
                pass

    return None, None


class Teacher:
    def __init__(self, model_dir: Path, device: torch.device):
        dtype = torch.float16 if device.type == "cuda" else torch.float32

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
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.02,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


def candidate_text(candidates: list[Candidate]) -> str:
    lines = []
    for candidate in candidates:
        details = []
        if candidate.source:
            details.append(f"source={candidate.source}")
        if candidate.target:
            details.append(f"target={candidate.target}")
        if candidate.relation:
            details.append(f"relation={candidate.relation}")
        suffix = f" | {', '.join(details)}" if details else ""
        lines.append(
            f"{candidate.candidate_id}: {candidate.action}{suffix}"
        )
    return "\n".join(lines)


def teacher_prompt(
    state: State,
    goal: dict,
    candidates: list[Candidate],
    refinement: bool = False,
    first_candidate: Candidate | None = None,
    first_result: State | None = None,
    first_status: dict | None = None,
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
    candidate_text_value = candidate_text(candidates)

    if not refinement:
        return [
            {
                "role": "system",
                "content": (
                    "You are a cognitive planning teacher. "
                    "Choose exactly one action from the supplied valid candidates. "
                    "The objective is explicit, but the preferred action is not. "
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
                    "Choose the single action that directly satisfies the "
                    "objective with the least unnecessary change."
                ),
            },
        ]

    assert first_result is not None
    assert first_status is not None

    return [
        {
            "role": "system",
            "content": (
                "You are a cognitive planning teacher reviewing your own "
                "previous decision. The first decision may be wrong. "
                "Use the validator feedback and choose ONE candidate that "
                "best satisfies the objective. Return ONLY JSON like "
                '{"candidate_id":0,"confidence":0.0} '
                "or a single line such as 0: BIND."
            ),
        },
        {
            "role": "user",
            "content": (
                "ORIGINAL WORKING MEMORY:\n"
                f"{state_text}\n\n"
                "GOAL:\n"
                f"{goal_text}\n\n"
                "YOUR FIRST CHOICE:\n"
                f"{json.dumps(candidate_dict(first_candidate), ensure_ascii=False)}\n\n"
                "STATE AFTER FIRST CHOICE:\n"
                f"{json.dumps(first_result.signature(), indent=2, ensure_ascii=False)}\n\n"
                "VALIDATOR FEEDBACK:\n"
                f"{json.dumps(first_status, ensure_ascii=False)}\n\n"
                "VALID CANDIDATES FOR REVISION:\n"
                f"{candidate_text_value}\n\n"
                "Choose the candidate that best satisfies the stated objective. "
                "Avoid NOOP unless the objective is already satisfied."
            ),
        },
    ]


def load_cached_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def balanced_quotas(total: int, actions: tuple[str, ...]) -> dict[str, int]:
    if total < len(actions):
        raise ValueError(
            f"--cases must be at least {len(actions)} so every action gets "
            "at least one sample."
        )
    base, remainder = divmod(total, len(actions))
    quotas = {action: base for action in actions}
    for action in actions[:remainder]:
        quotas[action] += 1
    return quotas


def build_seed_pool(
    db: ConceptNet,
    seed: int,
    pool_size: int,
) -> list[str]:
    rows = db.conn.execute(
        """
        SELECT start
        FROM edge
        WHERE relation IN (
            'IsA', 'CapableOf', 'HasProperty', 'UsedFor', 'HasA',
            'PartOf', 'RelatedTo', 'SimilarTo', 'Antonym', 'Causes',
            'AtLocation'
        )
        GROUP BY start
        HAVING COUNT(*) >= 2
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (max(pool_size, 1000),),
    ).fetchall()

    values = [row["start"] for row in rows if row["start"]]
    values = list(dict.fromkeys(values))
    random.Random(seed).shuffle(values)
    return values


def accepted_counts(path: Path) -> Counter:
    counter = Counter()
    for row in load_cached_records(path):
        action = row.get("requested_action")
        if action in ACTIONS:
            counter[action] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate action-balanced LLM teacher trajectories. "
            "The default is 500 accepted samples, balanced across all "
            "seven action categories."
        )
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=500,
        help="Number of ACCEPTED samples to generate (default: 500).",
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
        default=2101,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum teacher attempts. Default is max(cases*20, 2000). "
            "This prevents an infinite loop if one class becomes impossible."
        ),
    )
    args = parser.parse_args()

    if args.cases < len(ACTIONS):
        raise SystemExit(
            f"--cases must be >= {len(ACTIONS)}; got {args.cases}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== V210 ACTION-BALANCED TEACHER DISTILLATION ===", flush=True)
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

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

    quotas = balanced_quotas(args.cases, ACTIONS)
    accepted = accepted_counts(ACCEPTED_PATH)

    print("target_total:", args.cases, flush=True)
    print(
        "target_per_action:",
        json.dumps(quotas, sort_keys=True),
        flush=True,
    )
    print(
        "already_accepted:",
        json.dumps(
            {action: accepted[action] for action in ACTIONS},
            sort_keys=True,
        ),
        flush=True,
    )

    if all(accepted[action] >= quotas[action] for action in ACTIONS):
        print("Target dataset already exists; nothing to do.", flush=True)
        return

    db = ConceptNet(DB_PATH)
    teacher = Teacher(MODEL_DIR, device)
    rng = random.Random(args.seed)

    # Reuse already-written case IDs so a restarted run doesn't append a
    # duplicate attempt for the same ID.
    existing_raw = load_cached_records(RAW_PATH)
    next_case_number = len(existing_raw) + 1

    max_attempts = args.max_attempts or max(args.cases * 20, 2000)

    # Build a large seed pool once. We draw and reshuffle it repeatedly, so
    # every action class gets varied semantic neighborhoods.
    seed_pool = build_seed_pool(
        db,
        args.seed,
        pool_size=max(5000, args.cases * 8),
    )
    if not seed_pool:
        raise RuntimeError("ConceptNet returned no usable scenario seeds.")

    stats = Counter()
    started = time.perf_counter()

    try:
        with RAW_PATH.open("a", encoding="utf-8") as raw_file, ACCEPTED_PATH.open(
            "a",
            encoding="utf-8",
        ) as accepted_file:

            for attempt in range(1, max_attempts + 1):
                remaining = [
                    action
                    for action in ACTIONS
                    if accepted[action] < quotas[action]
                ]

                if not remaining:
                    break

                # Choose among the classes with the most remaining quota.
                max_remaining = max(
                    quotas[action] - accepted[action]
                    for action in remaining
                )
                weighted = [
                    action
                    for action in remaining
                    if quotas[action] - accepted[action] == max_remaining
                ]
                requested_action = rng.choice(weighted)
                seed = rng.choice(seed_pool)

                case_id = f"scenario_{next_case_number:06d}"
                next_case_number += 1

                stats["attempts"] += 1

                try:
                    state, goal = build_action_scenario(
                        db,
                        seed,
                        requested_action,
                        rng,
                        args.max_neighbors,
                    )
                    candidates = make_candidates(db, state, goal)

                    if not candidates:
                        stats["no_candidates"] += 1
                        continue

                    before_status = goal_status(state, goal)

                    # Turn 1.
                    first_raw = teacher.generate(
                        teacher_prompt(state, goal, candidates),
                        args.max_new_tokens,
                    )
                    first_id, first_confidence = parse_teacher(first_raw)
                    first_candidate = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate.candidate_id == first_id
                        ),
                        None,
                    )

                    first_valid = False
                    first_validation_reason = "no_candidate"
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

                    first_status = goal_status(first_state, goal)

                    # Turn 2 only when the objective is not yet satisfied.
                    refinement_needed = not first_status["goal_reached"]
                    second_raw = None
                    second_id = None
                    second_confidence = None
                    second_candidate = None
                    second_valid = False
                    second_validation_reason = "not_needed"
                    final_state = first_state.clone()

                    if refinement_needed:
                        second_raw = teacher.generate(
                            teacher_prompt(
                                state,
                                goal,
                                candidates,
                                refinement=True,
                                first_candidate=first_candidate,
                                first_result=first_state,
                                first_status=first_status,
                            ),
                            args.max_new_tokens,
                        )
                        second_id, second_confidence = parse_teacher(second_raw)
                        second_candidate = next(
                            (
                                candidate
                                for candidate in candidates
                                if candidate.candidate_id == second_id
                            ),
                            None,
                        )

                        if second_candidate is not None:
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

                    final_status = goal_status(final_state, goal)

                    final_action = (
                        second_candidate
                        if second_candidate is not None
                        else first_candidate
                    )

                    first_success = bool(first_status["goal_reached"])
                    final_success = bool(final_status["goal_reached"])

                    # Critical: acceptance is BOTH goal success AND the intended
                    # action family. This guarantees that the final dataset
                    # actually contains the requested action categories rather
                    # than merely solving the objectives through another action.
                    action_matches = (
                        final_action is not None
                        and final_action.action == requested_action
                    )
                    accepted_case = final_success and action_matches
                    corrected = (
                        not first_success
                        and accepted_case
                    )

                    if first_success:
                        stats["first_success"] += 1
                    if final_success:
                        stats["final_success"] += 1
                    if corrected:
                        stats["corrected"] += 1
                    if first_candidate is not None:
                        stats["first_parsed"] += 1
                    if second_candidate is not None:
                        stats["second_parsed"] += 1

                    record = {
                        "case_id": case_id,
                        "seed": seed,
                        "requested_action": requested_action,
                        "goal": goal,
                        "initial_state": state.signature(),
                        "candidates": [
                            candidate_dict(candidate)
                            for candidate in candidates
                        ],
                        "turn1": {
                            "raw_output": first_raw,
                            "parsed_candidate_id": first_id,
                            "confidence": first_confidence,
                            "candidate": candidate_dict(first_candidate),
                            "valid": first_valid,
                            "validation_reason": first_validation_reason,
                            "state_after": first_state.signature(),
                            "goal_status": first_status,
                        },
                        "turn2": {
                            "refinement_needed": refinement_needed,
                            "raw_output": second_raw,
                            "parsed_candidate_id": second_id,
                            "confidence": second_confidence,
                            "candidate": candidate_dict(second_candidate),
                            "valid": second_valid,
                            "validation_reason": second_validation_reason,
                            "state_after": final_state.signature(),
                            "goal_status": final_status,
                        },
                        "final_action": candidate_dict(final_action),
                        "first_success": first_success,
                        "final_success": final_success,
                        "action_matches_requested": action_matches,
                        "accepted": accepted_case,
                        "corrected": corrected,
                    }

                    raw_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    raw_file.flush()

                    if accepted_case:
                        accepted[requested_action] += 1
                        accepted_record = {
                            "case_id": case_id,
                            "seed": seed,
                            "requested_action": requested_action,
                            "goal": goal,
                            "initial_state": state.signature(),
                            "candidates": [
                                candidate_dict(candidate)
                                for candidate in candidates
                            ],
                            "teacher_turn1": {
                                "candidate": candidate_dict(first_candidate),
                                "confidence": first_confidence,
                                "success": first_success,
                            },
                            "teacher_turn2": {
                                "candidate": candidate_dict(second_candidate),
                                "confidence": second_confidence,
                                "used": refinement_needed,
                                "corrected": corrected,
                            },
                            "final_action": candidate_dict(final_action),
                            "final_state": final_state.signature(),
                            "action_sequence": [
                                candidate_dict(candidate)
                                for candidate in (
                                    [first_candidate]
                                    if first_success
                                    else [first_candidate, second_candidate]
                                )
                                if candidate is not None
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

                        stats["accepted"] += 1

                    if (
                        stats["attempts"] <= 5
                        or stats["attempts"] % 25 == 0
                    ):
                        remaining_text = ", ".join(
                            f"{action}={accepted[action]}/{quotas[action]}"
                            for action in ACTIONS
                        )
                        print(
                            f"ATTEMPT {stats['attempts']}/{max_attempts} "
                            f"accepted={stats['accepted']} "
                            f"last={requested_action}:{'YES' if accepted_case else 'NO'} "
                            f"| {remaining_text}",
                            flush=True,
                        )

                except Exception as exc:
                    stats["scenario_errors"] += 1
                    raw_file.write(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "seed": seed,
                                "requested_action": requested_action,
                                "error": repr(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raw_file.flush()

            complete = all(
                accepted[action] >= quotas[action]
                for action in ACTIONS
            )

            summary = {
                "experiment": "V210 action-balanced teacher distillation",
                "requested_accepted_cases": args.cases,
                "balanced_quotas": quotas,
                "accepted_counts": {
                    action: accepted[action]
                    for action in ACTIONS
                },
                "total_accepted": sum(
                    accepted[action] for action in ACTIONS
                ),
                "all_actions_represented": all(
                    accepted[action] > 0
                    for action in ACTIONS
                ),
                "quota_complete": complete,
                "attempts": stats["attempts"],
                "first_success_rate": (
                    stats["first_success"]
                    / max(1, stats["attempts"])
                ),
                "final_goal_success_rate": (
                    stats["final_success"]
                    / max(1, stats["attempts"])
                ),
                "accepted_rate": (
                    stats["accepted"]
                    / max(1, stats["attempts"])
                ),
                "correction_rate": (
                    stats["corrected"]
                    / max(1, stats["attempts"])
                ),
                "first_parse_rate": (
                    stats["first_parsed"]
                    / max(1, stats["attempts"])
                ),
                "second_parse_rate": (
                    stats["second_parsed"]
                    / max(1, stats["attempts"])
                ),
                "scenario_errors": stats["scenario_errors"],
                "raw_path": str(RAW_PATH.resolve()),
                "accepted_path": str(ACCEPTED_PATH.resolve()),
                "elapsed_seconds": time.perf_counter() - started,
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
            print("=== V210 COMPLETE ===")
            print("quota_complete:", complete)
            print("total_accepted:", summary["total_accepted"])
            print(
                "accepted_counts:",
                json.dumps(summary["accepted_counts"], sort_keys=True),
            )
            print("attempts:", summary["attempts"])
            print("accepted_rate:", summary["accepted_rate"])
            print("raw_cache:", RAW_PATH.resolve())
            print("accepted_dataset:", ACCEPTED_PATH.resolve())
            print("summary:", SUMMARY_PATH.resolve())

            if not complete:
                raise RuntimeError(
                    "Could not reach all action quotas within --max-attempts. "
                    "Increase --max-attempts if necessary."
                )

    finally:
        db.close()


if __name__ == "__main__":
    main()
