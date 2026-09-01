from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v632_semantic_core import (
    Graph,
    Hypothesis,
    LLMRealizer,
    Memory,
    SpaCyParser,
    relation_hypotheses,
    search,
    remember_success,
)


class PriorAdapter:
    """Adapt persistent memory path weights to the search controller API."""

    def __init__(self, memory):
        self.memory = memory

    @property
    def values(self):
        values = {}

        for key, value in self.memory.path_values.items():
            parts = tuple(
                str(key).split("|")
            )

            if len(parts) < 2:
                continue

            goal = parts[0]
            next_relation = parts[-1]
            prefix = parts[1:-1]

            values[
                (
                    goal,
                    tuple(prefix),
                    next_relation,
                )
            ] = float(value)

        return values

    def rank(
        self,
        goal,
        prefix,
        relations,
    ):
        rows = []

        for relation in relations:
            key = (
                str(goal),
                tuple(prefix),
                str(relation),
            )
            rows.append(
                (
                    self.values.get(
                        key,
                        0.0,
                    ),
                    relation,
                )
            )

        rows.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return rows


def append_trace(
    path,
    trace,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                trace,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def make_topics(
    graph,
    limit=10,
):
    preferred = [
        (
            "dog",
            "What is a dog?",
        ),
        (
            "animal",
            "What is an animal?",
        ),
        (
            "house",
            "What is a house?",
        ),
        (
            "water",
            "What is water?",
        ),
        (
            "person",
            "What is a person?",
        ),
        (
            "food",
            "What is food?",
        ),
        (
            "bird",
            "What is a bird?",
        ),
        (
            "car",
            "What is a car?",
        ),
        (
            "tree",
            "What is a tree?",
        ),
        (
            "language",
            "What is language?",
        ),
    ]

    topics = []

    for word, question in preferred:
        row = graph.conn.execute(
            """
            SELECT node
            FROM nodes
            WHERE normalized=?
            LIMIT 1
            """,
            (word,),
        ).fetchone()

        if not row:
            continue

        edges = graph.outgoing(
            word,
            80,
        )

        if any(
            edge.relation
            in {
                "definition",
                "is_a",
                "has_part",
                "part_of",
                "capable_of",
                "used_for",
                "has_property",
                "related_to",
            }
            for edge in edges
        ):
            topics.append(
                {
                    "word": word,
                    "question": question,
                }
            )

        if len(topics) >= limit:
            return topics

    rows = graph.conn.execute(
        """
        SELECT node
        FROM nodes
        WHERE is_common=1
          AND normalized=node
        ORDER BY LENGTH(node),node
        LIMIT 1000
        """
    ).fetchall()

    templates = (
        (
            "is_a",
            "What kind of thing is {word}?",
        ),
        (
            "has_part",
            "What parts does {word} have?",
        ),
        (
            "part_of",
            "What is {word} a part of?",
        ),
        (
            "capable_of",
            "What can {word} do?",
        ),
        (
            "used_for",
            "What is {word} used for?",
        ),
    )

    seen = {
        topic["word"]
        for topic in topics
    }

    for row in rows:
        word = str(
            row["node"]
        )

        if word in seen:
            continue

        relation_set = {
            edge.relation
            for edge in graph.outgoing(
                word,
                80,
            )
        }

        for relation, template in templates:
            if relation not in relation_set:
                continue

            topics.append(
                {
                    "word": word,
                    "question": template.format(
                        word=word
                    ),
                }
            )
            seen.add(word)
            break

        if len(topics) >= limit:
            break

    return topics


def concept_graph_first(
    graph,
    parse,
    memory,
):
    if (
        not (
            parse.question == "WH_WHAT"
            and parse.root_lemma in {
                "be",
                "mean",
                "refer",
            }
        )
        or parse.entities
    ):
        return None

    concept_words = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "be",
        "a",
        "an",
        "the",
        "of",
    }

    mention = None

    for value in (
        parse.noun_chunks
        + parse.subjects
        + parse.objects
    ):
        words = [
            word
            for word in str(value).split()
            if word.lower()
            not in concept_words
        ]
        if words:
            mention = " ".join(words)
            break

    if not mention:
        return None

    resolution = graph.resolve_entity_alias(
        mention,
        8,
    )

    if resolution.get(
        "status"
    ) != "resolved":
        return {
            "mention": mention,
            "resolution": resolution,
            "resolved": False,
        }

    memory.active_subject = resolution[
        "canonical"
    ]

    return {
        "mention": mention,
        "resolution": resolution,
        "resolved": True,
    }


def handle_turn(
    question,
    graph,
    parser,
    memory,
    realizer,
    args,
):
    started = time.perf_counter()

    t0 = time.perf_counter()
    parse = parser.parse(
        question
    )
    parse_seconds = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()
    concept = concept_graph_first(
        graph,
        parse,
        memory,
    )
    concept_seconds = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()

    hs = relation_hypotheses(
        parse,
        graph,
        memory,
        max_n=args.max_hypotheses,
    )

    hypotheses_seconds = (
        time.perf_counter()
        - t0
    )

    # When concept graph-first succeeded, inject a deterministic relation
    # hypothesis for definition. This isn't a semantic answer; it simply gives
    # the structural "what is X" form a graph goal.
    if (
        concept
        and concept.get(
            "resolved"
        )
    ):
        concept_subject = concept[
            "resolution"
        ]["canonical"]

        definition_h = Hypothesis(
            concept_subject,
            "definition",
            "concept_lookup",
            1.20,
            {
                "concept_graph_first": True,
                "concept_resolution": concept[
                    "resolution"
                ],
            },
        )

        hs = [
            definition_h,
            *[
                hypothesis
                for hypothesis in hs
                if hypothesis.intent
                != "conversation"
            ],
        ][:args.max_hypotheses]

    timing_search = time.perf_counter()

    prior = PriorAdapter(
        memory
    )

    ranked = []

    for index, hypothesis in enumerate(
        hs
    ):
        result = search(
            graph,
            prior,
            hypothesis,
            budget=args.goal_budget,
            per_node=args.per_node,
            max_depth=args.max_depth,
            seed=args.seed + index,
        )

        score = hypothesis.lexical_score

        if result.get(
            "success",
            False,
        ):
            score += 4.0

            if result.get(
                "direct_proof",
                False,
            ):
                score += 2.0

        score += min(
            1.0,
            float(
                result.get(
                    "attention",
                    0,
                )
            ) / 4.0,
        )

        ranked.append(
            (
                score,
                hypothesis,
                result,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1].relation,
        )
    )

    search_seconds = (
        time.perf_counter()
        - timing_search
    )

    if ranked:
        _, selected, result = ranked[0]
    else:
        selected = Hypothesis(
            None,
            "",
            "conversation",
            0.0,
            {},
        )
        result = {
            "success": False,
            "intent_only": True,
            "steps": 0,
            "path": [],
            "target": None,
            "attention": 0,
            "exploration": 0,
        }

    t0 = time.perf_counter()

    resolution = (
        selected.evidence.get(
            "entity_resolution",
            {},
        )
        if isinstance(
            selected.evidence,
            dict,
        )
        else {}
    )

    if not isinstance(
        resolution,
        dict,
    ):
        resolution = {}

    unresolved = (
        selected.intent
        == "entity_unresolved"
        and bool(
            resolution.get(
                "mention"
            )
        )
    )

    if result.get(
        "success",
        False,
    ):
        definition = graph.definition(
            selected.subject
        )

        answer = realizer.generate(
            realizer.grounded_prompt(
                question,
                selected,
                result,
                definition=definition,
            ),
            temperature=0.10,
        )
        mode = "grounded"

        if not answer:
            answer = (
                "The verified result is "
                + str(
                    result.get(
                        "target",
                    )
                )
                + "."
            )
    elif unresolved:
        answer = (
            "I couldn't verify that concept "
            "in the semantic graph."
        )
        mode = "unresolved"
    else:
        answer = realizer.generate(
            realizer.conversation_prompt(
                question,
                memory.turns,
            ),
            temperature=0.15,
        )
        mode = "conversation"

    realization_seconds = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()

    remember_success(
        memory,
        selected,
        result,
        parse,
    )

    learning_seconds = (
        time.perf_counter()
        - t0
    )

    total_seconds = (
        time.perf_counter()
        - started
    )

    route = {
        "intent": selected.intent,
        "subject": selected.subject,
        "relation": selected.relation,
        "mode": mode,
        "success": bool(
            result.get(
                "success",
                False,
            )
        ),
        "direct_proof": bool(
            result.get(
                "direct_proof",
                False,
            )
        ),
        "steps": result.get(
            "steps",
            0,
        ),
        "path": result.get(
            "path",
            [],
        ),
        "target": result.get(
            "target"
        ),
        "attention": result.get(
            "attention",
            0,
        ),
        "exploration": result.get(
            "exploration",
            0,
        ),
        "hypotheses": len(hs),
        "concept_graph_first": bool(
            concept
        ),
        "concept_resolved": bool(
            concept
            and concept.get(
                "resolved"
            )
        ),
    }

    trace = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "route": route,
        "parse": asdict(parse),
        "selected": asdict(selected),
        "hypotheses": [
            asdict(
                hypothesis
            )
            for _, hypothesis, _
            in ranked
        ],
        "search": result,
        "concept": concept,
        "timing": {
            "parse_seconds": parse_seconds,
            "concept_seconds": concept_seconds,
            "hypotheses_seconds": hypotheses_seconds,
            "search_seconds": search_seconds,
            "learning_seconds": learning_seconds,
            "realization_seconds": realization_seconds,
            "total_seconds": total_seconds,
        },
        "memory": {
            "turns": len(
                memory.turns
            ),
            "active_subject": memory.active_subject,
        },
    }

    memory.turns.append(
        trace
    )
    memory.turns = memory.turns[-256:]
    memory.save()

    return answer, trace


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--database",
        required=True,
    )
    ap.add_argument(
        "--output",
        default="",
    )
    ap.add_argument(
        "--trace-output",
        default="",
    )
    ap.add_argument(
        "--memory-output",
        default="",
    )
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
    )
    ap.add_argument(
        "--llm-model",
        default=r"C:\Users\adria\Desktop\dev\Graph-Topology\llm\SmolLM3-3B",
    )
    ap.add_argument(
        "--mode",
        choices=(
            "chat",
            "smoke",
        ),
        default="chat",
    )
    ap.add_argument(
        "--max-hypotheses",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--goal-budget",
        type=int,
        default=40,
    )
    ap.add_argument(
        "--per-node",
        type=int,
        default=60,
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--cache-entries",
        type=int,
        default=12000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=63200,
    )

    args = ap.parse_args()

    database = Path(
        args.database
    ).resolve()

    base = Path(
        args.output
        if args.output
        else database.parent
        / "v632_chat.json"
    ).resolve()

    trace_path = Path(
        args.trace_output
        if args.trace_output
        else base.with_name(
            base.stem
            + "_traces.jsonl"
        )
    ).resolve()

    memory_path = Path(
        args.memory_output
        if args.memory_output
        else base.with_name(
            base.stem
            + "_memory.json"
        )
    ).resolve()

    print(
        "=== V632 SEMANTIC CHAT ===",
        flush=True,
    )
    print(
        f"semantic database : {database}",
        flush=True,
    )
    print(
        "knowledge source  : WordNet + ConceptNet 5.7",
        flush=True,
    )
    print(
        "vocabulary        : compact beginner dictionary",
        flush=True,
    )
    print(
        "grammar           : frozen spaCy",
        flush=True,
    )
    print(
        "search            : conditional prior + BFS",
        flush=True,
    )
    print(
        f"memory            : {memory_path}",
        flush=True,
    )

    graph = Graph(
        database,
        args.cache_entries,
    )

    parser = SpaCyParser(
        args.spacy_model
    )

    memory = Memory(
        memory_path
    )

    realizer = LLMRealizer(
        args.llm_model
    )

    topics = make_topics(
        graph,
        10,
    )

    print(
        "\nTOPICS / EXAMPLES",
        flush=True,
    )

    for index, topic in enumerate(
        topics,
        1,
    ):
        print(
            f"  {index}. "
            f"{topic['question']}",
            flush=True,
        )

    print(
        "\nCommands: help, exit",
        flush=True,
    )

    if args.mode == "smoke":
        questions = [
            "What is a dog?",
            "What is an animal?",
            "What can a dog do?",
            "What parts does a dog have?",
            "What is a house?",
        ]
    else:
        questions = None

    if questions is not None:
        for question in questions:
            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                realizer,
                args,
            )

            print(
                f"\nQ: {question}",
                flush=True,
            )
            print(
                f"A: {answer}",
                flush=True,
            )
            print(
                f"  mode={trace['route']['mode']} "
                f"route="
                + (
                    " -> ".join(
                        trace["route"]["path"]
                    )
                    if trace["route"]["path"]
                    else "conversation"
                ),
                flush=True,
            )
            print(
                f"  steps={trace['route']['steps']} "
                f"attention={trace['route']['attention']} "
                f"exploration={trace['route']['exploration']} "
                f"time={trace['timing']['total_seconds']:.3f}s",
                flush=True,
            )
            append_trace(
                trace_path,
                trace,
            )
    else:
        print(
            "\n=== CHAT ===",
            flush=True,
        )

        while True:
            try:
                question = input(
                    "chat> "
                ).strip()
            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
            }:
                break

            if question.lower() in {
                "help",
                "?",
            }:
                print(
                    "\nTOPICS / EXAMPLES",
                    flush=True,
                )
                for index, topic in enumerate(
                    topics,
                    1,
                ):
                    print(
                        f"  {index}. "
                        f"{topic['question']}",
                        flush=True,
                    )
                continue

            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                realizer,
                args,
            )

            route = trace["route"]
            timing = trace["timing"]

            print(
                f"answer: {answer}",
                flush=True,
            )
            print(
                f"  mode={route['mode']} "
                f"intent={route['intent']} "
                f"relation={route['relation']!r}",
                flush=True,
            )
            print(
                "  route="
                + (
                    " -> ".join(
                        route["path"]
                    )
                    if route["path"]
                    else "conversation"
                ),
                flush=True,
            )
            print(
                f"  result="
                f"{'VERIFIED' if route['success'] else 'NOT VERIFIED'} "
                f"steps={route['steps']} "
                f"direct={route['direct_proof']} "
                f"attention={route['attention']} "
                f"exploration={route['exploration']}",
                flush=True,
            )
            print(
                f"  time={timing['total_seconds']:.3f}s "
                f"(parse={timing['parse_seconds']:.3f}s "
                f"search={timing['search_seconds']:.3f}s "
                f"llm={timing['realization_seconds']:.3f}s)",
                flush=True,
            )

            append_trace(
                trace_path,
                trace,
            )

    print(
        "\n=== V632 COMPLETE ===",
        flush=True,
    )
    print(
        f"trace : {trace_path}",
        flush=True,
    )
    print(
        f"memory: {memory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
