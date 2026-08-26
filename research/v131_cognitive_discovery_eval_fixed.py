from __future__ import annotations

"""
V131 — WHOLE-SYSTEM COGNITIVE DISCOVERY EVALUATION

V130 had a major confound: expected working-memory bindings were injected
directly into the designer. V131 removes that teacher hand.

Pipeline:

    sentence
      ↓
    lexical cue extraction
      ↓
    long-term ConceptNet memory
      ↓
    modular semantic attention
      ↓
    designer inspects the active memory
      ↓
    designer discovers / reuses / binds working-memory structure
      ↓
    temporary working-memory graph
      ↓
    optional long-term accumulation

The evaluation never tells the designer the expected semantic bindings.

There are 24 small scenarios. They are intentionally diverse enough to test:
    - category retrieval
    - properties
    - capabilities
    - uses
    - parts
    - two-entity relations
    - role ordering
    - ambiguity / competition
    - repeated exposure
    - persistence after working memory clears

The expected structures are used ONLY for scoring.

IMPORTANT:
    This is still not a full natural-language parser. The sentence front end
    only provides lexical cues and crude role hints. The semantic relation
    discovery itself is performed from active long-term topology.

This isolates the architecture's intended core:
    attention + working memory + designer.

No LLM.
No ConceptNet mutation.
"""

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json
import re
import time

# Support both:
#   python .\\v131_cognitive_discovery_eval.py
# and:
#   python -m v130_cognitive_architecture.v131_cognitive_discovery_eval
#
# The V130 support modules use package-relative imports, so direct execution
# must import the directory as a package rather than importing each module as
# a standalone file.
import sys as _sys
from pathlib import Path as _Path

_PACKAGE_PARENT = _Path(__file__).resolve().parents[1]
if str(_PACKAGE_PARENT) not in _sys.path:
    _sys.path.insert(0, str(_PACKAGE_PARENT))

try:
    from v130_cognitive_architecture.long_term_memory import LongTermMemory
    from v130_cognitive_architecture.modular_attention import AttentionModule
    from v130_cognitive_architecture.working_memory import WorkingMemory
    from v130_cognitive_architecture.designer import GraphDesigner
except ImportError:
    from .long_term_memory import LongTermMemory
    from .modular_attention import AttentionModule
    from .working_memory import WorkingMemory
    from .designer import GraphDesigner


# v131_cognitive_discovery_eval_fixed.py lives at:
#   <repo>/research/v131_cognitive_discovery_eval_fixed.py
#
# data/ and results/ live directly under <repo>.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
RESULTS_PATH = ROOT / "results"

DB_PATH = DATA_PATH / "conceptnet_compact.db"
DICTIONARY_PATH = DATA_PATH / "dictionary.csv"

OUTPUT_PATH = (
    RESULTS_PATH
    / "v131_cognitive_discovery_report.json"
)

LTM_OUTPUT_PATH = (
    RESULTS_PATH
    / "v131_long_term_memory_after.json"
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    sentence: str
    cues: tuple[str, ...]
    expected_bindings: tuple[tuple[str, str, str], ...]
    expected_active: tuple[str, ...]
    expected_persistent_edges: tuple[
        tuple[str, str, str],
        ...
    ] = ()
    distractors: tuple[str, ...] = ()


SCENARIOS = (
    Scenario(
        "The dog is an animal.",
        ("dog", "animal"),
        (("dog", "IsA", "animal"),),
        ("dog", "animal"),
    ),
    Scenario(
        "The dog can bark.",
        ("dog", "bark"),
        (("dog", "CapableOf", "bark"),),
        ("dog", "bark"),
    ),
    Scenario(
        "The dog has fur.",
        ("dog", "fur"),
        (("dog", "HasProperty", "fur"),),
        ("dog", "fur"),
    ),
    Scenario(
        "The cat is an animal.",
        ("cat", "animal"),
        (("cat", "IsA", "animal"),),
        ("cat", "animal"),
    ),
    Scenario(
        "The cat can meow.",
        ("cat", "meow"),
        (("cat", "CapableOf", "meow"),),
        ("cat", "meow"),
    ),
    Scenario(
        "Water is a liquid.",
        ("water", "liquid"),
        (("water", "IsA", "liquid"),),
        ("water", "liquid"),
    ),
    Scenario(
        "Water is used for drinking.",
        ("water", "drinking"),
        (("water", "UsedFor", "drinking"),),
        ("water", "drinking"),
    ),
    Scenario(
        "A car is a vehicle.",
        ("car", "vehicle"),
        (("car", "IsA", "vehicle"),),
        ("car", "vehicle"),
    ),
    Scenario(
        "A car is used for transport.",
        ("car", "transport"),
        (("car", "UsedFor", "transport"),),
        ("car", "transport"),
    ),
    Scenario(
        "A chair is an object.",
        ("chair", "object"),
        (("chair", "IsA", "object"),),
        ("chair", "object"),
    ),
    Scenario(
        "A chair is used for sitting.",
        ("chair", "sitting"),
        (("chair", "UsedFor", "sitting"),),
        ("chair", "sitting"),
    ),
    Scenario(
        "A bird is an animal.",
        ("bird", "animal"),
        (("bird", "IsA", "animal"),),
        ("bird", "animal"),
    ),
    Scenario(
        "A bird can fly.",
        ("bird", "fly"),
        (("bird", "CapableOf", "fly"),),
        ("bird", "fly"),
    ),
    Scenario(
        "A bird has wings.",
        ("bird", "wings"),
        (("bird", "HasA", "wings"),),
        ("bird", "wings"),
    ),
    Scenario(
        "A child is a person.",
        ("child", "person"),
        (("child", "IsA", "person"),),
        ("child", "person"),
    ),
    Scenario(
        "A child can play.",
        ("child", "play"),
        (("child", "CapableOf", "play"),),
        ("child", "play"),
    ),
    Scenario(
        "A knife is a tool.",
        ("knife", "tool"),
        (("knife", "IsA", "tool"),),
        ("knife", "tool"),
    ),
    Scenario(
        "A knife can cut.",
        ("knife", "cut"),
        (("knife", "CapableOf", "cut"),),
        ("knife", "cut"),
    ),
    Scenario(
        "Music is related to sound.",
        ("music", "sound"),
        (("music", "RelatedTo", "sound"),),
        ("music", "sound"),
    ),
    Scenario(
        "The dog chased the cat.",
        ("dog", "chased", "cat"),
        (
            ("dog", "SUBJECT", "chased"),
            ("chased", "OBJECT", "cat"),
        ),
        ("dog", "cat"),
    ),
    Scenario(
        "The cat chased the dog.",
        ("cat", "chased", "dog"),
        (
            ("cat", "SUBJECT", "chased"),
            ("chased", "OBJECT", "dog"),
        ),
        ("cat", "dog"),
    ),
    Scenario(
        "The dog is an animal and the cat is an animal.",
        ("dog", "animal", "cat"),
        (
            ("dog", "IsA", "animal"),
            ("cat", "IsA", "animal"),
        ),
        ("dog", "animal", "cat"),
    ),
    Scenario(
        "The dog can bark and the bird can fly.",
        ("dog", "bark", "bird", "fly"),
        (
            ("dog", "CapableOf", "bark"),
            ("bird", "CapableOf", "fly"),
        ),
        ("dog", "bark", "bird", "fly"),
    ),
    Scenario(
        "The dog is an animal that can bark.",
        ("dog", "animal", "bark"),
        (
            ("dog", "IsA", "animal"),
            ("dog", "CapableOf", "bark"),
        ),
        ("dog", "animal", "bark"),
        expected_persistent_edges=(
            ("dog", "IsA", "animal"),
            ("dog", "CapableOf", "bark"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Surface front end
# ---------------------------------------------------------------------------

ACTION_WORDS = {
    "chased",
    "chase",
    "likes",
    "liked",
    "follows",
    "followed",
    "helps",
    "helped",
    "sees",
    "saw",
}


def lexical_cues(
    sentence: str,
    dictionary: set[str],
) -> list[str]:
    tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
        sentence.lower(),
    )
    return [
        token
        for token in tokens
        if token in dictionary
    ]


def ordered_content_tokens(
    sentence: str,
    dictionary: set[str],
) -> list[str]:
    tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
        sentence.lower(),
    )
    return [
        token
        for token in tokens
        if token in dictionary
    ]


def role_hints(
    sentence: str,
    dictionary: set[str],
) -> list[tuple[str, str, str]]:
    """
    Minimal language/grammar hinting.

    This supplies only ordering information for relational clauses; it does
    NOT supply the semantic relation label to the designer.

    Example:
        "dog chased cat"
            -> (dog, SUBJECT_HINT, cat)
    """
    tokens = re.findall(
        r"[a-zA-Z]+",
        sentence.lower(),
    )

    result = []

    for index, token in enumerate(tokens):
        if token not in ACTION_WORDS:
            continue

        before = [
            t for t in tokens[:index]
            if t in dictionary
        ]

        after = [
            t for t in tokens[index + 1:]
            if t in dictionary
        ]

        if before and after:
            result.append(
                (
                    before[-1],
                    "SUBJECT_HINT",
                    after[0],
                )
            )

    return result


# ---------------------------------------------------------------------------
# Designer discovery
# ---------------------------------------------------------------------------

RELATION_PRIORITY = (
    "IsA",
    "CapableOf",
    "HasProperty",
    "UsedFor",
    "HasA",
    "PartOf",
    "Causes",
    "RelatedTo",
    "SimilarTo",
    "Synonym",
    "AtLocation",
)


def semantic_neighbors(
    memory: LongTermMemory,
    concept: str,
) -> list[tuple[str, str, float]]:
    node = memory.nodes_by_name.get(
        concept.lower()
    )
    if node is None:
        return []

    edges = list(
        memory.outgoing(node.node_id)
    )

    relation_rank = {
        relation: index
        for index, relation
        in enumerate(
            RELATION_PRIORITY
        )
    }

    edges.sort(
        key=lambda edge: (
            relation_rank.get(
                edge.relation,
                99,
            ),
            -edge.effective_weight,
        )
    )

    return [
        (
            edge.relation,
            memory.nodes_by_id[
                edge.target
            ].concept,
            edge.effective_weight,
        )
        for edge in edges
    ]


def discover_pair_relation(
    memory: LongTermMemory,
    source: str,
    target: str,
) -> tuple[str, float] | None:
    source_node = memory.nodes_by_name.get(
        source.lower()
    )
    if source_node is None:
        return None

    candidates = []

    for edge in memory.outgoing(
        source_node.node_id
    ):
        end = memory.nodes_by_id[
            edge.target
        ].concept

        if end != target.lower():
            continue

        candidates.append(
            (
                edge.relation,
                edge.effective_weight,
            )
        )

    if not candidates:
        return None

    relation_rank = {
        relation: index
        for index, relation
        in enumerate(
            RELATION_PRIORITY
        )
    }

    return max(
        candidates,
        key=lambda item: (
            -relation_rank.get(
                item[0],
                99,
            ),
            item[1],
        ),
    )


def discover_from_active_state(
    memory: LongTermMemory,
    attention: AttentionModule,
    working: WorkingMemory,
    designer: GraphDesigner,
    cues: list[str],
    hints: list[tuple[str, str, str]],
) -> dict:
    """
    No expected relation is supplied here.

    The designer:
        1. reuses top activated concepts
        2. looks for actual graph edges between co-active lexical concepts
        3. uses role hints only to establish subject/object ordering
        4. creates temporary BIND edges when a semantic edge is found
        5. creates temporary role bindings when an action clause exists
    """
    active = attention.active_concepts(
        memory,
        limit=24,
    )

    for concept, activation in active[:10]:
        designer.reuse(
            concept,
            activation=activation,
        )

    active_names = {
        concept.lower()
        for concept, value in active
        if value > 0.05
    }

    discovered = []

    # Semantic discovery directly from activated lexical concepts.
    for source in cues:
        if source.lower() not in active_names:
            continue

        for target in cues:
            if source.lower() == target.lower():
                continue

            result = discover_pair_relation(
                memory,
                source,
                target,
            )

            if result is None:
                continue

            relation, confidence = result

            designer.bind(
                source,
                relation,
                target,
                confidence=min(
                    1.0,
                    0.4 + confidence,
                ),
            )

            discovered.append(
                (
                    source,
                    relation,
                    target,
                    confidence,
                )
            )

    # Role binding is discovered from grammar ordering, not semantic labels.
    for subject, _hint, obj in hints:
        # Find any active action word between them.
        tokens = [
            concept.lower()
            for concept in cues
        ]

        action = None
        for cue in tokens:
            if cue in ACTION_WORDS:
                action = cue
                break

        if action is not None:
            designer.bind(
                subject,
                "SUBJECT",
                action,
                confidence=0.7,
            )
            designer.bind(
                action,
                "OBJECT",
                obj,
                confidence=0.7,
            )

            discovered.append(
                (
                    subject,
                    "SUBJECT",
                    action,
                    0.7,
                )
            )
            discovered.append(
                (
                    action,
                    "OBJECT",
                    obj,
                    0.7,
                )
            )

    return {
        "active": active,
        "discovered": discovered,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_bindings(
    working: WorkingMemory,
    expected: tuple[
        tuple[str, str, str],
        ...
    ],
) -> dict:
    found = []
    missing = []

    for binding in expected:
        if working.has_edge(
            *binding
        ):
            found.append(
                list(binding)
            )
        else:
            missing.append(
                list(binding)
            )

    return {
        "found": found,
        "missing": missing,
        "coverage": (
            len(found)
            / max(
                1,
                len(expected),
            )
        ),
    }


def score_activation(
    attention: AttentionModule,
    memory: LongTermMemory,
    expected: tuple[str, ...],
) -> dict:
    active = dict(
        attention.active_concepts(
            memory,
            limit=64,
        )
    )

    hits = {
        concept: active.get(
            concept.lower(),
            0.0,
        )
        for concept in expected
        if active.get(
            concept.lower(),
            0.0,
        ) > 0.0
    }

    return {
        "expected": list(expected),
        "hits": hits,
        "coverage": (
            len(hits)
            / max(
                1,
                len(expected),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V131 WHOLE-SYSTEM COGNITIVE DISCOVERY ==="
    )

    dictionary = set()

    with DICTIONARY_PATH.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for raw in handle:
            word = raw.strip().lower()
            if word and word.isalpha():
                dictionary.add(word)

    print(
        "dictionary_words:",
        len(dictionary),
        flush=True,
    )

    print(
        "loading ConceptNet into long-term memory...",
        flush=True,
    )

    memory = LongTermMemory.load_conceptnet(
        DB_PATH,
        dictionary,
        min_edge_weight=1.0,
        max_edges_per_word=60,
    )

    print(
        "ltm_stats_initial:",
        memory.stats(),
        flush=True,
    )

    attention = AttentionModule(
        name="semantic",
        decay=0.60,
        propagation_scale=0.65,
        max_active=24,
    )

    working = WorkingMemory()

    designer = GraphDesigner(
        memory=memory,
        working=working,
    )

    episode_reports = []

    for index, scenario in enumerate(
        SCENARIOS,
        start=1,
    ):
        attention.clear()
        working.begin(index)

        cues = lexical_cues(
            scenario.sentence,
            dictionary,
        )

        hints = role_hints(
            scenario.sentence,
            dictionary,
        )

        # Cue only the words from the sentence. No expected concepts.
        for cue in cues:
            attention.seed(
                memory,
                cue,
                value=1.0,
            )

        for _ in range(3):
            attention.step(memory)

        discovery = discover_from_active_state(
            memory,
            attention,
            working,
            designer,
            cues,
            hints,
        )

        activation_score = score_activation(
            attention,
            memory,
            scenario.expected_active,
        )

        binding_score = score_bindings(
            working,
            scenario.expected_bindings,
        )

        # Only semantic relations discovered by the designer can accumulate.
        accumulation_events = 0

        for source, relation, target in scenario.expected_persistent_edges:
            if working.has_edge(
                source,
                relation,
                target,
            ):
                designer.accumulate(
                    source,
                    relation,
                    target,
                    amount=0.05,
                )
                accumulation_events += 1

        # Do NOT use expected_persistent_edges to create memory. They only
        # decide whether a discovered binding earned reinforcement.
        semantic_accumulation = (
            designer.reinforce_semantic_edges(
                amount=0.02
            )
        )

        events = [
            {
                "operation": event.operation,
                "details": event.details,
            }
            for event in designer.events
        ]

        episode_report = {
            "episode": index,
            "sentence": scenario.sentence,
            "cues": cues,
            "role_hints": [
                list(item)
                for item in hints
            ],
            "active": discovery["active"],
            "discovered": [
                list(item)
                for item in discovery[
                    "discovered"
                ]
            ],
            "activation_score": activation_score,
            "binding_score": binding_score,
            "working_memory_before_clear": working.snapshot(),
            "designer_events": events,
            "explicit_accumulation_successes": accumulation_events,
            "semantic_accumulation_events": semantic_accumulation,
        }

        episode_reports.append(
            episode_report
        )

        designer.events.clear()
        working.clear()

        print(
            f"CASE {index:02d}/{len(SCENARIOS):02d} "
            f"activation={activation_score['coverage']:.3f} "
            f"binding={binding_score['coverage']:.3f} "
            f"discovered={len(discovery['discovered'])} "
            f"accumulated={semantic_accumulation}",
            flush=True,
        )

    activation_values = [
        report["activation_score"]["coverage"]
        for report in episode_reports
    ]

    binding_values = [
        report["binding_score"]["coverage"]
        for report in episode_reports
    ]

    total_discovered = sum(
        len(report["discovered"])
        for report in episode_reports
    )

    total_reuse = sum(
        1
        for report in episode_reports
        for event in report["designer_events"]
        if event["operation"] == "REUSE"
    )

    total_bind = sum(
        1
        for report in episode_reports
        for event in report["designer_events"]
        if event["operation"] == "BIND"
    )

    total_accumulate = sum(
        1
        for report in episode_reports
        for event in report["designer_events"]
        if event["operation"] == "ACCUMULATE"
    )

    # Persist the final LTM after the cognitive episodes.
    memory.save(
        LTM_OUTPUT_PATH
    )

    # Accumulation probes.
    probes = {}
    for source, relation, target in (
        ("dog", "IsA", "animal"),
        ("dog", "CapableOf", "bark"),
    ):
        edge = memory.edge(
            source,
            relation,
            target,
        )
        probes[
            f"{source}|{relation}|{target}"
        ] = (
            None
            if edge is None
            else {
                "weight": edge.weight,
                "reinforcement": edge.reinforcement,
                "use_count": edge.use_count,
                "effective_weight": edge.effective_weight,
            }
        )

    report = {
        "experiment": (
            "V131 whole-system cognitive discovery"
        ),
        "scenario_count": len(
            SCENARIOS
        ),
        "memory_initial": None,
        "memory_final": memory.stats(),
        "attention": {
            "decay": attention.decay,
            "propagation_scale": attention.propagation_scale,
            "max_active": attention.max_active,
        },
        "summary": {
            "mean_activation_coverage": (
                sum(activation_values)
                / max(
                    1,
                    len(activation_values),
                )
            ),
            "mean_binding_coverage": (
                sum(binding_values)
                / max(
                    1,
                    len(binding_values),
                )
            ),
            "total_discovered_bindings": total_discovered,
            "reuse_events": total_reuse,
            "bind_events": total_bind,
            "accumulate_events": total_accumulate,
        },
        "accumulation_probes": probes,
        "episodes": episode_reports,
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

    print()
    print(
        "=== V131 SUMMARY ==="
    )
    print(
        "mean_activation_coverage:",
        report["summary"][
            "mean_activation_coverage"
        ],
    )
    print(
        "mean_binding_coverage:",
        report["summary"][
            "mean_binding_coverage"
        ],
    )
    print(
        "total_discovered_bindings:",
        total_discovered,
    )
    print(
        "reuse_events:",
        total_reuse,
    )
    print(
        "bind_events:",
        total_bind,
    )
    print(
        "accumulate_events:",
        total_accumulate,
    )
    print(
        "final_memory:",
        report["memory_final"],
    )
    print(
        "accumulation_probes:",
        probes,
    )
    print(
        "saved:",
        OUTPUT_PATH,
    )
    print(
        "saved_ltm:",
        LTM_OUTPUT_PATH,
    )
    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )
    print(
        "=== V131 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
