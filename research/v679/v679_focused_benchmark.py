#!/usr/bin/env python3
"""Run graph-grounded dialogue probes selected from the focused V679 graph."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from v679_semantic_chat_gateway import (
    LiveSemanticTeacher,
    LocalLLMRuntime,
    Realizer,
    WorkerDiscoveryReader,
    handle_turn,
)
from v679_semantic_core import Attention, Context, Graph, SpaCyParser
from v679_attention import AttentionController


class BenchmarkMemory(Context):
    """Ephemeral conversation state: benchmark runs do not mutate chat memory."""

    def save(self):
        return None


class BenchmarkDistilled:
    """Disable learned decisions so every case exposes the current routing."""

    def lookup(self, *args, **kwargs):
        return None

    def learn(self, *args, **kwargs):
        return None

    def goal_learn(self, *args, **kwargs):
        return None


CASE_SPECS = (
    ("definition", "definition", 3),
    ("part", "has_a", 10),
    ("property", "has_property", 8),
    ("type", "is_a", 8),
    ("capability", "capable_of", 3),
)


def direct_facts(graph, relation, count):
    rows = graph.conn.execute(
        """
        SELECT e.subject, e.relation, e.object,
               COALESCE(n.label, e.object) AS label
        FROM edges e
        LEFT JOIN nodes n ON n.node=e.object
        WHERE e.subject IN ('en:dog', 'en:bear', 'en:animal')
          AND e.relation=?
        ORDER BY e.subject, lower(COALESCE(n.label, e.object)), e.object
        LIMIT ?
        """,
        (relation, count),
    ).fetchall()
    if len(rows) != count:
        raise RuntimeError(
            f"Focused graph has {len(rows)} usable {relation} facts; "
            f"benchmark requires {count}."
        )
    return [dict(row) for row in rows]


def question_for(goal, subject, label):
    if goal == "part":
        return f"Does {subject} have {label}?"
    if goal == "property":
        return f"Is {subject} {label}?"
    if goal == "type":
        return f"Is {subject} a {label}?"
    if goal == "capability":
        return f"Can {subject} {label}?"
    raise ValueError(f"Unsupported goal: {goal}")


def build_cases(graph):
    cases = [
        {
            "id": f"definition_{subject}",
            "group": "definition",
            "question": (
                f"What is {'an' if subject[0] in 'aeiou' else 'a'} {subject}?"
            ),
            "expected": {
                "subject": f"en:{subject}",
                "semantic_goal": "definition",
                "direct_proof": True,
            },
        }
        for subject in ("animal", "bear", "dog")
    ]

    for goal, relation, count in CASE_SPECS[1:]:
        for fact in direct_facts(graph, relation, count):
            subject = graph.node_label(fact["subject"])
            label = str(fact["label"])
            cases.append(
                {
                    "id": f"{goal}_{subject}_{fact['object']}".replace(":", "_"),
                    "group": goal,
                    "question": question_for(goal, subject, label),
                    "expected": {
                        "subject": fact["subject"],
                        "semantic_goal": goal,
                        "relation": relation,
                        "target": fact["object"],
                        "target_label": label,
                        "direct_proof": True,
                    },
                }
            )
    return cases


def normalization_variants(cases):
    variants = []
    for case in cases:
        question = case["question"]
        if question.startswith("What is "):
            variant = question.replace("What is a ", "What is the ", 1)
            variant = variant.replace("What is an ", "What is the ", 1)
        elif question.startswith("Does dog "):
            variant = question.replace("Does dog ", "Does the dog ", 1)
        elif question.startswith("Does bear "):
            variant = question.replace("Does bear ", "Does the bear ", 1)
        elif question.startswith("Does animal "):
            variant = question.replace("Does animal ", "Does an animal ", 1)
        elif question.startswith("Is dog "):
            variant = question.replace("Is dog ", "Is the dog ", 1)
        elif question.startswith("Is bear "):
            variant = question.replace("Is bear ", "Is the bear ", 1)
        elif question.startswith("Is animal "):
            variant = question.replace("Is animal ", "Is an animal ", 1)
        elif question.startswith("Can animal "):
            variant = question.replace("Can animal ", "Can an animal ", 1)
        else:
            raise ValueError(f"No normalization variant for {question!r}")
        variants.append({
            **case,
            "id": case["id"] + "_determiner",
            "question": variant,
            "normalization_variant": "optional_determiner",
            "source_case_id": case["id"],
        })
    return variants


def grammar_normalization_cases():
    cases = []
    for subject in ("animal", "bear", "dog"):
        cases.append({
            "id": f"definition_{subject}_case_punctuation",
            "group": "definition",
            "question": f"WHAT IS {'AN' if subject[0] in 'aeiou' else 'A'} {subject.upper()}!!!",
            "normalization_variant": "case_punctuation",
            "expected": {
                "subject": f"en:{subject}",
                "semantic_goal": "definition",
                "direct_proof": True,
            },
        })
        cases.append({
            "id": f"definition_{subject}_plural",
            "group": "definition",
            "question": f"What are {subject}s?",
            "normalization_variant": "regular_plural",
            "expected": {
                "subject": f"en:{subject}",
                "semantic_goal": "definition",
                "direct_proof": True,
            },
        })
    cases.append({
        "id": "property_dog_larger_than_a_cat",
        "group": "property",
        "question": "Is a dog larger than a cat?",
        "normalization_variant": "internal_optional_determiner",
        "expected": {
            "subject": "en:dog",
            "semantic_goal": "property",
            "relation": "has_property",
            "target": "en:larger than cat",
            "target_label": "larger than cat",
            "direct_proof": True,
        },
    })
    cases.append({
        "id": "capability_animal_eat_inflection",
        "group": "capability",
        "question": "Can an animal eat?",
        "normalization_variant": "verb_inflection",
        "expected": {
            "subject": "en:animal",
            "semantic_goal": "capability",
            "relation": "capable_of",
            "target": "en:eating",
            "target_label": "eating",
            "direct_proof": True,
        },
    })
    cases.extend((
        {
            "id": "part_dog_plural_subject",
            "group": "part",
            "question": "Do dogs have teeth?",
            "normalization_variant": "plural_subject_auxiliary",
            "expected": {
                "subject": "en:dog",
                "semantic_goal": "part",
                "relation": "has_a",
                "target": "en:teeth",
                "target_label": "teeth",
                "direct_proof": True,
            },
        },
        {
            "id": "property_dog_plural_subject",
            "group": "property",
            "question": "Are dogs brown?",
            "normalization_variant": "plural_subject_copular",
            "expected": {
                "subject": "en:dog",
                "semantic_goal": "property",
                "relation": "has_property",
                "target": "en:brown",
                "target_label": "brown",
                "direct_proof": True,
            },
        },
        {
            "id": "capability_animal_plural_subject",
            "group": "capability",
            "question": "Can animals eat?",
            "normalization_variant": "plural_subject_inflection",
            "expected": {
                "subject": "en:animal",
                "semantic_goal": "capability",
                "relation": "capable_of",
                "target": "en:eating",
                "target_label": "eating",
                "direct_proof": True,
            },
        },
    ))
    for subject in ("animal", "bear", "dog"):
        cases.append({
            "id": f"definition_{subject}_contraction",
            "group": "definition",
            "question": f"What's {'an' if subject[0] in 'aeiou' else 'a'} {subject}?",
            "normalization_variant": "interrogative_contraction",
            "expected": {
                "subject": f"en:{subject}",
                "semantic_goal": "definition",
                "direct_proof": True,
            },
        })
    for subject in ("dog", "bear", "animal"):
        expected = {
            "subject": f"en:{subject}",
            "semantic_goal": "part",
            "direct_proof": True,
        }
        for index, question in enumerate((
            f"What parts does {subject} have?",
            f"What are the parts of {subject}?",
            f"What are {subject}'s parts?",
            f"What's {subject}'s parts?",
        ), 1):
            cases.append({
                "id": f"part_inventory_{subject}_{index}",
                "group": "part_inventory",
                "question": question,
                "normalization_variant": "part_subject_placement",
                "expected": expected,
            })
        cases.append({
            "id": f"part_inventory_{subject}_terse",
            "group": "part_inventory",
            "question": f"parts of an {subject}" if subject[0] in "aeiou" else f"parts of a {subject}",
            "normalization_variant": "terse_prepositional_fragment",
            "expected": expected,
        })
    return cases


def worker_discovery_cases(shared_memory):
    """Use ten focused, live worker discoveries without inventing static facts."""
    reader = WorkerDiscoveryReader(shared_memory)
    discoveries = reader.discoveries(limit=10)
    return [
        {
            "id": f"worker_discovery_{discovery['kind']}_{discovery['subject']}".replace(":", "_"),
            "group": "worker_discovery",
            "question": reader.topic_for(discovery),
            "expected": {
                "subject": discovery["subject"],
                "semantic_goal": "type",
                "target": discovery["object"],
                "worker_record_key": discovery["key"],
            },
        }
        for discovery in discoveries
    ]


def main():
    ap = argparse.ArgumentParser(
        description="Run focused graph dialogue benchmark cases."
    )
    ap.add_argument("--database", required=True)
    ap.add_argument("--output", default="./results/v679/benchmark.jsonl")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--max-hypotheses", type=int, default=12)
    ap.add_argument("--goal-budget", type=int, default=40)
    ap.add_argument("--per-node", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache-entries", type=int, default=12000)
    ap.add_argument(
        "--normalization-variants",
        action="store_true",
        help="Run determiner, contraction, and part-subject grammar variants.",
    )
    ap.add_argument(
        "--shared-memory",
        default="",
        help="Include live worker-discovery transition cases from this checkpoint.",
    )
    ap.add_argument(
        "--adversarial-ambiguity",
        action="store_true",
        help="Also emit controller-only ambiguity and abstention benchmark cases.",
    )
    args = ap.parse_args()

    graph = Graph(args.database, args.cache_entries)
    cases = build_cases(graph)
    if args.normalization_variants:
        cases += normalization_variants(cases)
        cases += grammar_normalization_cases()
    worker_discoveries = (
        WorkerDiscoveryReader(args.shared_memory)
        if args.shared_memory
        else None
    )
    if worker_discoveries is not None:
        cases += worker_discovery_cases(args.shared_memory)
    if args.adversarial_ambiguity:
        cases += [{
            "id": "adversarial_no_valid_answer",
            "group": "adversarial_ambiguity",
            "question": "Is a dog a nonexistent target?",
            "expected": {
                "semantic_decision_outcome": "abstain",
                "route_success": False,
            },
        }]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parser = SpaCyParser(args.spacy_model)
    runtime = LocalLLMRuntime(args.llm_model)
    teacher = LiveSemanticTeacher(runtime, temperature=0.05)
    realizer = Realizer(runtime)
    distilled = BenchmarkDistilled()

    with output.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            memory = BenchmarkMemory()
            answer, trace = handle_turn(
                case["question"],
                graph,
                parser,
                memory,
                Attention(0.65),
                realizer,
                teacher,
                distilled,
                args,
                worker_discoveries,
                controller=AttentionController(),
            )
            route = trace["route"]
            expected = case["expected"]
            if "semantic_decision_outcome" in expected:
                passed = (
                    trace["semantic_decision"]["outcome"]
                    == expected["semantic_decision_outcome"]
                    and route["success"] == expected["route_success"]
                )
            elif expected.get("intent") == "worker_discovery":
                passed = (
                    route["success"]
                    and route["intent"] == "worker_discovery"
                    and route["subject"] == expected["subject"]
                    and "worker_discovery" in trace
                    and trace["worker_discovery"]["kind"] == expected["kind"]
                    and trace["worker_discovery"]["record_key"] == expected["record_key"]
                )
            else:
                passed = (
                    route["success"]
                    and route["subject"] == expected["subject"]
                    and (
                        expected["semantic_goal"]
                        == trace["search"].get("semantic_goal", route["relation"])
                    )
                    and (
                        "target" not in expected
                        or route["target"] == expected["target"]
                    )
                )
                if "direct_proof" in expected:
                    passed = passed and route["direct_proof"] == expected["direct_proof"]
            record = {
                "record_type": "benchmark_case",
                "benchmark": "v679_focused_graph",
                "case_index": index,
                **case,
                "answer": answer,
                "passed": passed,
                "instrumentation": trace,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:02d}/{len(cases):02d}] "
                f"{'PASS' if passed else 'FAIL'} {case['question']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
