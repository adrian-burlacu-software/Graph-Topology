from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import sqlite3

from v633_full_semantic_core import (
    Graph,
    Context,
    Attention,
    Hypothesis,
    SpaCyParser,
    relation_hypotheses,
    search,
)


def append_trace(
    path,
    payload,
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
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


class MemoryContext(Context):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return
        try:
            payload = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return
        self.active_subject = payload.get(
            "active_subject"
        )
        self.turns = payload.get(
            "turns",
            [],
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.path.write_text(
            json.dumps(
                {
                    "active_subject": self.active_subject,
                    "turns": self.turns[-256:],
                    "entities": self.entities,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


class Realizer:
    def __init__(
        self,
        model_path,
    ):
        self.model_path = str(
            model_path
        )
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                device_map="auto",
            )
        )

    def generate(
        self,
        prompt,
        temperature=0.10,
        max_new_tokens=96,
    ):
        self.load()

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        device = getattr(
            self.model,
            "device",
            None,
        )
        if device is not None:
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

        import torch

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature
                if temperature > 0.0
                else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = encoded[
            "input_ids"
        ].shape[1]

        return self.tokenizer.decode(
            output[
                0,
                prompt_len:,
            ],
            skip_special_tokens=True,
        ).strip().split(
            "\n",
            1,
        )[0].strip()

    def grounded_prompt(
        self,
        question,
        hypothesis,
        result,
        definition=None,
    ):
        evidence = " -> ".join(
            result.get(
                "path",
                [],
            )
        )

        return (
            "You are a semantic surface realizer.\n"
            "The graph result below is authoritative.\n"
            "Use only the verified result and supplied evidence.\n"
            "Do not add outside facts.\n"
            "Answer one concise natural sentence.\n\n"
            f"QUESTION: {question}\n"
            f"SUBJECT: {hypothesis.subject}\n"
            f"RELATION: {hypothesis.relation}\n"
            f"VERIFIED RESULT: {result.get('target')}\n"
            f"EVIDENCE PATH: {evidence}\n"
            + (
                f"DEFINITION: {definition}\n"
                if definition
                else ""
            )
            + "ANSWER:"
        )

    def conversation_prompt(
        self,
        question,
        history,
    ):
        recent = []

        for turn in history[-6:]:
            if not isinstance(
                turn,
                dict,
            ):
                continue

            recent.append(
                "USER: "
                + str(
                    turn.get(
                        "question",
                        turn.get(
                            "text",
                            "",
                        ),
                    )
                )[:240]
            )
            recent.append(
                "ASSISTANT: "
                + str(
                    turn.get(
                        "answer",
                        "",
                    )
                )[:280]
            )

        return (
            "You are the conversational assistant.\n"
            "Be natural and concise.\n"
            "You may greet the user, tell jokes, "
            "and explain general concepts.\n"
            "Do not claim that the semantic graph "
            "verified facts it did not verify.\n\n"
            "RECENT:\n"
            + (
                "\n".join(recent)
                if recent
                else "none"
            )
            + f"\nUSER: {question}\nASSISTANT:"
        )


def choose_best(
    ranked,
):
    if not ranked:
        return (
            Hypothesis(
                None,
                "",
                "conversation",
                0.0,
                {},
            ),
            {
                "success": False,
                "steps": 0,
                "path": [],
                "target": None,
                "attention": 0,
                "exploration": 0,
            },
        )

    ranked.sort(
        key=lambda row: (
            -row[0],
            row[1].relation,
        )
    )

    return (
        ranked[0][1],
        ranked[0][2],
    )


def handle_turn(
    question,
    graph,
    parser,
    memory,
    attention,
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
    hypotheses = relation_hypotheses(
        parse,
        graph,
        memory,
        max_n=args.max_hypotheses,
    )
    hypothesis_seconds = (
        time.perf_counter()
        - t0
    )

    t0 = time.perf_counter()

    ranked = []

    for index, hypothesis in enumerate(
        hypotheses
    ):
        result = search(
            graph,
            attention,
            hypothesis,
            budget=args.goal_budget,
            per_node=args.per_node,
            max_depth=args.max_depth,
        )

        score = (
            hypothesis.lexical_score
            + (
                5.0
                if result.get(
                    "success",
                    False,
                )
                else 0.0
            )
            + (
                1.0
                if result.get(
                    "direct_proof",
                    False,
                )
                else 0.0
            )
        )

        ranked.append(
            (
                score,
                hypothesis,
                result,
            )
        )

    search_seconds = (
        time.perf_counter()
        - t0
    )

    selected, result = choose_best(
        ranked
    )

    # Resolve conversational fallback only after the semantic attempt.
    t0 = time.perf_counter()

    entity_resolution = (
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
        entity_resolution,
        dict,
    ):
        entity_resolution = {}

    if (
        result.get(
            "success",
            False,
        )
    ):
        definition = graph.definition(
            selected.subject
        )

        answer = realizer.generate(
            realizer.grounded_prompt(
                question,
                selected,
                result,
                definition,
            ),
            temperature=0.10,
        )
        mode = "grounded"
    elif (
        selected.intent
        == "entity_unresolved"
    ):
        answer = (
            "I couldn't verify that entity "
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

    llm_seconds = (
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
        "path": list(
            result.get(
                "path",
                [],
            )
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
    }

    trace = {
        "timestamp": time.time(),
        "question": question,
        "answer": answer,
        "route": route,
        "entity_resolution": entity_resolution,
        "parse": asdict(parse),
        "selected": asdict(selected),
        "hypotheses": [
            asdict(
                h
            )
            for _, h, _
            in ranked[:args.max_hypotheses]
        ],
        "search": result,
        "timing": {
            "parse_seconds": parse_seconds,
            "hypothesis_seconds": hypothesis_seconds,
            "search_seconds": search_seconds,
            "llm_seconds": llm_seconds,
            "total_seconds": total_seconds,
        },
    }

    memory.active_subject = selected.subject
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
        required=True,
    )
    ap.add_argument(
        "--memory-output",
        required=True,
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
        choices=("chat","smoke"),
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
        default=63300,
    )

    args = ap.parse_args()

    database = Path(
        args.database
    ).resolve()
    trace_path = Path(
        args.trace_output
    ).resolve()
    memory_path = Path(
        args.memory_output
    ).resolve()

    print(
        "=== V633 FULL SEMANTIC CHAT ===",
        flush=True,
    )
    print(
        f"database : {database}",
        flush=True,
    )
    stats_probe = time.perf_counter()
    try:
        with sqlite3.connect(
            str(database),
            timeout=30.0,
        ) as stats_conn:
            node_count = stats_conn.execute(
                "SELECT COUNT(*) FROM nodes"
            ).fetchone()[0]
            edge_count = stats_conn.execute(
                "SELECT COUNT(*) FROM edges"
            ).fetchone()[0]
        print(
            f"graph stats: nodes={node_count:,} edges={edge_count:,} "
            f"probe={time.perf_counter()-stats_probe:.3f}s",
            flush=True,
        )
    except Exception as exc:
        print(
            f"graph stats: unavailable ({exc})",
            flush=True,
        )
    print(
        "knowledge: ALL WordNet + ALL English ConceptNet",
        flush=True,
    )
    print(
        "grammar  : frozen spaCy",
        flush=True,
    )
    print(
        "search   : contextual path attention + bounded BFS",
        flush=True,
    )

    graph = Graph(
        database,
        args.cache_entries,
    )
    parser = SpaCyParser(
        args.spacy_model
    )

    memory = MemoryContext(
        memory_path
    )
    memory.load()

    attention = Attention(
        0.65
    )

    realizer = Realizer(
        args.llm_model
    )

    print(
        "\nTOPICS / EXAMPLES",
        flush=True,
    )

    topics = [
        "What is a dog?",
        "What is an animal?",
        "What is a house?",
        "What is water?",
        "What is a person?",
        "What can a dog do?",
        "What parts does a dog have?",
        "What is a bird?",
        "What is a tree?",
        "What is food?",
    ]

    for index, question in enumerate(
        topics,
        1,
    ):
        print(
            f"  {index}. {question}",
            flush=True,
        )

    print(
        "\nCommands: help, exit",
        flush=True,
    )

    if args.mode == "smoke":
        questions = topics[:7]
    else:
        questions = None

    if questions:
        for question in questions:
            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                attention,
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
                f"  result="
                f"{'VERIFIED' if trace['route']['success'] else 'NOT VERIFIED'} "
                f"steps={trace['route']['steps']} "
                f"attention={trace['route']['attention']} "
                f"exploration={trace['route']['exploration']}",
                flush=True,
            )
            print(
                f"  time={trace['timing']['total_seconds']:.3f}s "
                f"(search={trace['timing']['search_seconds']:.3f}s "
                f"llm={trace['timing']['llm_seconds']:.3f}s)",
                flush=True,
            )
            append_trace(
                trace_path,
                trace,
            )
    else:
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
                for index, item in enumerate(
                    topics,
                    1,
                ):
                    print(
                        f"  {index}. {item}",
                        flush=True,
                    )
                continue

            answer, trace = handle_turn(
                question,
                graph,
                parser,
                memory,
                attention,
                realizer,
                args,
            )

            print(
                f"answer: {answer}",
                flush=True,
            )
            print(
                f"  mode={trace['route']['mode']} "
                f"intent={trace['route']['intent']} "
                f"relation={trace['route']['relation']!r}",
                flush=True,
            )
            print(
                "  route="
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
                f"  result="
                f"{'VERIFIED' if trace['route']['success'] else 'NOT VERIFIED'} "
                f"steps={trace['route']['steps']} "
                f"attention={trace['route']['attention']} "
                f"exploration={trace['route']['exploration']}",
                flush=True,
            )
            print(
                f"  time={trace['timing']['total_seconds']:.3f}s "
                f"(search={trace['timing']['search_seconds']:.3f}s "
                f"llm={trace['timing']['llm_seconds']:.3f}s)",
                flush=True,
            )

            append_trace(
                trace_path,
                trace,
            )

    print(
        "\n=== V633 COMPLETE ===",
        flush=True,
    )
    print(
        f"TRACE  : {trace_path}",
        flush=True,
    )
    print(
        f"MEMORY : {memory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
