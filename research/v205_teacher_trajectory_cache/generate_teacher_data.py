from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from .trajectory_schema import (
        ACTIONS,
        ACTION_TO_ID,
        CandidateAction,
        WorkingState,
        candidate_to_dict,
    )
except ImportError:
    import sys

    _RESEARCH_ROOT = (
        Path(__file__).resolve().parents[1]
    )

    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(_RESEARCH_ROOT),
        )

    from v205_teacher_trajectory_cache.trajectory_schema import (
        ACTIONS,
        ACTION_TO_ID,
        CandidateAction,
        WorkingState,
        candidate_to_dict,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LLM = ROOT / "llm"
RESULTS = ROOT / "results"

DB_PATH = DATA / "conceptnet_compact.db"
MODEL_DIR = LLM / "SmolLM2-1.7B-Instruct"

RAW_CACHE = (
    RESULTS
    / "v205_teacher_trajectories.jsonl"
)

ACCEPTED_CACHE = (
    RESULTS
    / "v205_teacher_dataset.jsonl"
)


# ---------------------------------------------------------------------------
# Relations
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

RELATION_SET = set(
    RELATIONS
)


# ---------------------------------------------------------------------------
# Teacher prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a cognitive planning teacher.

You are given a temporary working-memory graph and a numbered list of actions
that are ALREADY VALID.

Choose the single candidate action that is most useful as the next cognitive
step.

Do not invent an action.
Do not invent a node.
Do not invent a relation.
Return ONLY JSON:

{"candidate_id": 0, "confidence": 0.0}

candidate_id must be one of the supplied candidate IDs.
confidence must be between 0 and 1.
"""


# ---------------------------------------------------------------------------
# Graph memory
# ---------------------------------------------------------------------------

class ConceptNet:
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.conn = sqlite3.connect(
            str(path)
        )
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def source_edges(
        self,
        source: str,
        limit: int = 10,
    ) -> list[tuple[str, str, str, float]]:
        rows = self.conn.execute(
            """
            SELECT start, relation, end, weight
            FROM edge
            WHERE start = ?
            ORDER BY weight DESC
            LIMIT ?
            """,
            (
                source,
                limit,
            ),
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

    def has_edge(
        self,
        source: str,
        relation: str,
        target: str,
    ) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM edge
            WHERE start = ?
              AND relation = ?
              AND end = ?
            LIMIT 1
            """,
            (
                source,
                relation,
                target,
            ),
        ).fetchone()

        return row is not None


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def build_working_state(
    db: ConceptNet,
    seed: str,
    *,
    max_neighbors: int = 6,
) -> WorkingState:
    edges = db.source_edges(
        seed,
        limit=max_neighbors,
    )

    nodes = [
        {
            "concept": seed,
            "activation": 1.0,
            "role": 1,
            "persistent": False,
        }
    ]

    seen = {seed}

    state = WorkingState(
        nodes=[],
        edges=[],
    )

    state.add_node(
        seed,
        activation=1.0,
        role=1,
    )

    for source, relation, target, _weight in edges:
        if target in seen:
            continue

        seen.add(target)

        # First neighbor is an inactive semantic target; later neighbors are
        # weakly active distractors.
        activation = (
            0.0
            if len(seen) == 2
            else 0.35
        )

        state.add_node(
            target,
            activation=activation,
            role=2 if len(seen) == 2 else 3,
        )

        state.add_edge(
            source,
            relation,
            target,
            activation=0.15,
        )

    return state


def make_candidates(
    db: ConceptNet,
    state: WorkingState,
) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    next_id = 0

    def add(
        action: str,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> None:
        nonlocal next_id

        candidates.append(
            CandidateAction(
                candidate_id=next_id,
                action=action,
                source=source,
                target=target,
                relation=relation,
            )
        )
        next_id += 1

    add("NOOP")

    # REUSE: inactive existing concept.
    for node in state.nodes:
        if (
            node.activation < 0.5
            and node.concept
            != state.nodes[0].concept
        ):
            add(
                "REUSE",
                target=node.concept,
            )

    # BIND: only semantic edges known to the long-term graph. Prefer currently
    # inactive targets to create meaningful working-memory transitions.
    for edge in state.edges:
        target_node = state.node(
            edge.target
        )
        if target_node is None:
            continue

        if state.has_edge(
            edge.source,
            edge.relation,
            edge.target,
            active_only=True,
        ):
            continue

        if db.has_edge(
            edge.source,
            edge.relation,
            edge.target,
        ):
            add(
                "BIND",
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
            )

    # INHIBIT: active non-seed nodes.
    for node in state.nodes:
        if (
            node.concept
            != state.nodes[0].concept
            and node.activation > 0.5
        ):
            add(
                "INHIBIT",
                target=node.concept,
            )

    # BRANCH: source concept + known relation.
    if state.nodes:
        source = state.nodes[0].concept
        for edge in state.edges[:3]:
            add(
                "BRANCH",
                source=source,
                relation=edge.relation,
            )

    # Generic actions.
    add("CREATE")

    if any(
        node.activation > 0.5
        for node in state.nodes
    ) or any(
        edge.activation > 0.5
        for edge in state.edges
    ):
        add("COMMIT")

    # Deduplicate exact candidate tuples.
    unique = []
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
        unique.append(
            candidate
        )

    for index, candidate in enumerate(
        unique
    ):
        unique[index] = CandidateAction(
            candidate_id=index,
            action=candidate.action,
            source=candidate.source,
            target=candidate.target,
            relation=candidate.relation,
        )

    return unique


# ---------------------------------------------------------------------------
# Environment validation/application
# ---------------------------------------------------------------------------

def validate_candidate(
    db: ConceptNet,
    state: WorkingState,
    candidate: CandidateAction,
) -> tuple[bool, str]:
    action = candidate.action

    if action not in ACTION_TO_ID:
        return False, "unknown_action"

    if action == "NOOP":
        return True, "ok"

    if action == "CREATE":
        return True, "ok"

    if action == "COMMIT":
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

    if action == "REUSE":
        if candidate.target is None:
            return False, "missing_target"

        node = state.node(
            candidate.target
        )

        if node is None:
            return False, "unknown_target"

        return True, "ok"

    if action == "INHIBIT":
        if candidate.target is None:
            return False, "missing_target"

        node = state.node(
            candidate.target
        )

        if node is None:
            return False, "unknown_target"

        if node.activation <= 0.5:
            return False, "target_not_active"

        return True, "ok"

    if action == "BRANCH":
        if (
            candidate.source is None
            or candidate.relation is None
        ):
            return False, "missing_branch_arg"

        if state.node(
            candidate.source
        ) is None:
            return False, "unknown_source"

        if candidate.relation not in RELATION_SET:
            return False, "unknown_relation"

        return True, "ok"

    if action == "BIND":
        if (
            candidate.source is None
            or candidate.target is None
            or candidate.relation is None
        ):
            return False, "missing_bind_arg"

        if (
            state.node(
                candidate.source
            )
            is None
            or state.node(
                candidate.target
            )
            is None
        ):
            return False, "unknown_node"

        if candidate.relation not in RELATION_SET:
            return False, "unknown_relation"

        if not db.has_edge(
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


def apply_candidate(
    state: WorkingState,
    candidate: CandidateAction,
) -> WorkingState:
    next_state = state.clone()
    action = candidate.action

    if action == "NOOP":
        return next_state

    if action == "CREATE":
        next_state.add_node(
            f"created_{len(next_state.nodes)}",
            activation=0.85,
            role=6,
        )
        return next_state

    if action == "REUSE":
        assert candidate.target is not None
        node = next_state.node(
            candidate.target
        )
        if node is not None:
            node.activation = 1.0
        return next_state

    if action == "INHIBIT":
        assert candidate.target is not None
        node = next_state.node(
            candidate.target
        )
        if node is not None:
            node.activation *= 0.05
        return next_state

    if action == "BRANCH":
        assert candidate.source is not None
        assert candidate.relation is not None

        concept = (
            candidate.source
            + "#branch"
            + str(
                len(next_state.nodes)
            )
        )
        next_state.add_node(
            concept,
            activation=0.8,
            role=7,
        )
        next_state.add_edge(
            candidate.source,
            candidate.relation,
            concept,
            activation=0.8,
        )
        return next_state

    if action == "BIND":
        assert candidate.source is not None
        assert candidate.target is not None
        assert candidate.relation is not None

        source = next_state.node(
            candidate.source
        )
        target = next_state.node(
            candidate.target
        )

        if source is not None:
            source.activation = max(
                source.activation,
                1.0,
            )

        if target is not None:
            target.activation = max(
                target.activation,
                1.0,
            )

        next_state.add_edge(
            candidate.source,
            candidate.relation,
            candidate.target,
            activation=1.0,
        )
        return next_state

    if action == "COMMIT":
        for node in next_state.nodes:
            if node.activation > 0.5:
                node.persistent = True

        for edge in next_state.edges:
            if edge.activation > 0.5:
                edge.persistent = True

        return next_state

    raise ValueError(
        f"Unknown action: {action}"
    )


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

class Teacher:
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
    ) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"SmolLM2 directory not found: {model_dir.resolve()}"
            )

        dtype = (
            torch.float16
            if device.type == "cuda"
            else torch.float32
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                local_files_only=True,
            )
        )

        self.model.to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.device = device

    def prompt(
        self,
        state: WorkingState,
        candidates: list[CandidateAction],
    ) -> str:
        state_json = json.dumps(
            state.signature(),
            ensure_ascii=False,
            indent=2,
        )

        candidate_json = json.dumps(
            [
                candidate_to_dict(
                    candidate
                )
                for candidate in candidates
            ],
            ensure_ascii=False,
            indent=2,
        )

        return f"""The long-term semantic memory is authoritative.

Current temporary working memory:
{state_json}

Choose the single most useful next cognitive action from the VALID candidates
below. Do not invent candidates.

Valid candidates:
{candidate_json}

Return ONLY:
{{"candidate_id": 0, "confidence": 0.0}}
"""

    def generate(
        self,
        state: WorkingState,
        candidates: list[CandidateAction],
        max_new_tokens: int,
    ) -> tuple[str, str]:
        user = self.prompt(
            state,
            candidates,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cognitive planning teacher. "
                    "Choose only among supplied candidate IDs. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": user,
            },
        ]

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
            prompt = (
                messages[0]["content"]
                + "\n"
                + messages[1]["content"]
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

        raw = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        return user, raw


def parse_teacher(
    raw: str,
) -> tuple[int | None, float | None]:
    raw = raw.strip()

    try:
        value = json.loads(
            raw
        )
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")

        if (
            start < 0
            or end <= start
        ):
            return None, None

        try:
            value = json.loads(
                raw[start : end + 1]
            )
        except json.JSONDecodeError:
            return None, None

    if not isinstance(
        value,
        dict,
    ):
        return None, None

    candidate_id = value.get(
        "candidate_id"
    )
    confidence = value.get(
        "confidence"
    )

    try:
        candidate_id = int(
            candidate_id
        )
    except (
        TypeError,
        ValueError,
    ):
        candidate_id = None

    try:
        confidence = float(
            confidence
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

    return candidate_id, confidence


# ---------------------------------------------------------------------------
# Cached generation
# ---------------------------------------------------------------------------

def load_existing_ids() -> set[str]:
    if not RAW_CACHE.exists():
        return set()

    done = set()

    with RAW_CACHE.open(
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

            case_id = row.get(
                "case_id"
            )

            if case_id:
                done.add(
                    case_id
                )

    return done


def build_case_ids(
    db: ConceptNet,
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    conn = db.conn

    rows = conn.execute(
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
            count * 3,
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
    )[:count]

    if len(seeds) < count:
        raise RuntimeError(
            "Could not construct enough teacher scenarios."
        )

    return [
        (
            f"scenario_{index:05d}",
            seed,
        )
        for index, seed
        in enumerate(
            seeds,
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
        default=205,
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
        "=== V205 VALIDATED TEACHER TRAJECTORY CACHE ===",
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

    try:
        teacher = Teacher(
            MODEL_DIR,
            device,
        )

        existing = load_existing_ids()

        scenarios = build_case_ids(
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

        with RAW_CACHE.open(
            "a",
            encoding="utf-8",
        ) as raw_file, ACCEPTED_CACHE.open(
            "a",
            encoding="utf-8",
        ) as accepted_file:

            for index, (
                case_id,
                seed,
            ) in enumerate(
                scenarios,
                1,
            ):
                if case_id in existing:
                    continue

                state = build_working_state(
                    db,
                    seed,
                    max_neighbors=args.max_neighbors,
                )

                candidates = make_candidates(
                    db,
                    state,
                )

                if not candidates:
                    stats[
                        "no_candidates"
                    ] += 1
                    continue

                prompt, raw = teacher.generate(
                    state,
                    candidates,
                    args.max_new_tokens,
                )

                candidate_id, confidence = (
                    parse_teacher(
                        raw
                    )
                )

                parsed = (
                    candidate_id is not None
                )

                selected = None
                validation_reason = (
                    "no_selection"
                )

                if parsed:
                    for candidate in candidates:
                        if (
                            candidate.candidate_id
                            == candidate_id
                        ):
                            selected = candidate
                            break

                    if selected is None:
                        validation_reason = (
                            "candidate_id_not_in_menu"
                        )

                valid = False
                next_state = None

                if selected is not None:
                    valid, validation_reason = (
                        validate_candidate(
                            db,
                            state,
                            selected,
                        )
                    )

                    if valid:
                        next_state = (
                            apply_candidate(
                                state,
                                selected,
                            )
                        )

                record = {
                    "case_id": case_id,
                    "seed": seed,
                    "state": state.signature(),
                    "candidates": [
                        candidate_to_dict(
                            candidate
                        )
                        for candidate
                        in candidates
                    ],
                    "prompt": prompt,
                    "raw_output": raw,
                    "parsed_candidate_id": candidate_id,
                    "teacher_confidence": confidence,
                    "selected_candidate": (
                        candidate_to_dict(
                            selected
                        )
                        if selected is not None
                        else None
                    ),
                    "parsed": parsed,
                    "valid": valid,
                    "validation_reason": (
                        validation_reason
                    ),
                    "next_state": (
                        next_state.signature()
                        if next_state is not None
                        else None
                    ),
                }

                raw_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()

                if valid and selected is not None:
                    accepted_record = {
                        **record,
                        "teacher_action": (
                            candidate_to_dict(
                                selected
                            )
                        ),
                        "action_id": ACTION_TO_ID[
                            selected.action
                        ],
                        "next_state": next_state.signature(),
                    }

                    accepted_file.write(
                        json.dumps(
                            accepted_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted_file.flush()

                    stats[
                        "accepted"
                    ] += 1

                else:
                    stats[
                        "rejected"
                    ] += 1

                stats[
                    "parsed"
                ] += int(
                    parsed
                )

                existing.add(
                    case_id
                )

                processed = (
                    stats[
                        "accepted"
                    ]
                    + stats[
                        "rejected"
                    ]
                )

                if (
                    processed <= 5
                    or processed % 25 == 0
                    or processed == args.cases
                ):
                    acceptance = (
                        stats[
                            "accepted"
                        ]
                        / max(
                            1,
                            processed,
                        )
                    )

                    print(
                        f"TEACHER "
                        f"{processed}/{args.cases} "
                        f"parsed={stats['parsed']} "
                        f"accepted={stats['accepted']} "
                        f"rejected={stats['rejected']} "
                        f"acceptance={acceptance:.3f}",
                        flush=True,
                    )

        processed = (
            stats["accepted"]
            + stats["rejected"]
        )

        print()
        print(
            "=== V205 COMPLETE ==="
        )
        print(
            "processed_this_run:",
            processed,
        )
        print(
            "parsed:",
            stats["parsed"],
        )
        print(
            "accepted:",
            stats["accepted"],
        )
        print(
            "rejected:",
            stats["rejected"],
        )
        print(
            "acceptance:",
            (
                stats["accepted"]
                / max(
                    1,
                    processed,
                )
            ),
        )
        print(
            "raw_cache:",
            RAW_CACHE.resolve(),
        )
        print(
            "accepted_dataset:",
            ACCEPTED_CACHE.resolve(),
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
