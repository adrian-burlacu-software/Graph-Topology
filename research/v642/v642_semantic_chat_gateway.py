from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
import sqlite3

from v642_semantic_core import (
    Graph,
    DistilledMemory,
    candidate_senses,
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

        relation_wording = {
            "definition": "definition or meaning",
            "is_a": "type or category",
            "has_part": "parts or components",
            "part_of": "larger thing it belongs to",
            "capable_of": "ability or action",
            "used_for": "purpose or use",
            "has_property": "property or characteristic",
            "at_location": "location",
            "related_to": "related concept",
            "causes": "cause or effect",
            "made_of": "material or substance",
            "has_a": "possessed or contained thing",
        }.get(
            hypothesis.relation,
            hypothesis.relation.replace(
                "_",
                " ",
            ),
        )

        return (
            "You are a semantic surface realizer.\n"
            "The semantic graph is authoritative.\n"
            "Use ONLY the verified result below.\n"
            "Do not add outside facts.\n"
            "Do not change the relation.\n"
            "Return one concise natural sentence.\n\n"
            f"QUESTION: {question}\n"
            f"SUBJECT: {hypothesis.subject}\n"
            f"REQUESTED RELATION: {hypothesis.relation}\n"
            f"RELATION MEANING: {relation_wording}\n"
            f"PROOF KIND: {result.get('proof_kind')}\n"
            f"VERIFIED RESULT: {result.get('target')}\n"
            f"EVIDENCE PATH: {evidence}\n"
            + (
                f"DICTIONARY DEFINITION: {definition}\n"
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


class LiveSemanticTeacher:
    """
    Candidate-constrained SmolLM3 teacher.

    The model sees only the user question and a compact comma-separated list
    of graph-generated candidates. It must select exactly one supplied item.
    """

    def __init__(
        self,
        model_path,
        temperature=0.05,
    ):
        self.model_path = str(
            model_path
        )
        self.temperature = float(
            temperature
        )
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map="auto",
        )

    def choose(
        self,
        question,
        candidates,
        decision_type,
        descriptions=None,
    ):
        import json
        import re

        if not candidates:
            return None

        self._load()

        items = []
        for index, candidate in enumerate(
            candidates,
            1,
        ):
            item = {
                "id": index,
                "candidate": str(candidate),
            }

            if descriptions:
                item["description"] = str(
                    descriptions.get(
                        candidate,
                        "",
                    )
                )

            items.append(item)

        prompt = (
            "You are a semantic disambiguation teacher.\n"
            "Choose the single candidate that best matches the user's meaning.\n"
            "The candidates are authoritative and exhaustive for this decision.\n"
            "You MUST choose exactly one candidate from the list.\n"
            "Do not invent candidates.\n"
            "Return ONLY valid JSON in this exact shape:\n"
            '{"selected_id": 1, "confidence": 0.0}\n\n'
            "USER: "
            + str(question)
            + "\n"
            "DECISION_TYPE: "
            + str(decision_type)
            + "\n"
            "CANDIDATES_JSON:\n"
            + json.dumps(
                items,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            "JSON:"
        )

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
                max_new_tokens=32,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = encoded[
            "input_ids"
        ].shape[1]

        raw = self.tokenizer.decode(
            output[
                0,
                prompt_len:,
            ],
            skip_special_tokens=True,
        ).strip()

        # Extract a JSON object even if the small model adds a harmless prefix
        # or suffix. The decision is still accepted only if selected_id maps
        # to one of the graph-supplied candidates.
        match = re.search(
            r"\{.*?\}",
            raw,
            flags=re.DOTALL,
        )

        if not match:
            return None

        try:
            payload = json.loads(
                match.group(0)
            )
        except Exception:
            return None

        try:
            selected_id = int(
                payload["selected_id"]
            )
        except Exception:
            return None

        if not (
            1
            <= selected_id
            <= len(candidates)
        ):
            return None

        try:
            confidence = float(
                payload.get(
                    "confidence",
                    0.5,
                )
            )
        except Exception:
            confidence = 0.5

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return {
            "selected": str(
                candidates[
                    selected_id - 1
                ]
            ),
            "confidence": confidence,
            "raw": raw,
            "selected_id": selected_id,
            "format": "json",
        }



def distilled_choice(
    teacher,
    distilled,
    decision_type,
    surface,
    context_text,
    candidates,
    descriptions=None,
):
    if not candidates:
        return None, "none", 0.0

    remembered = distilled.lookup(
        decision_type,
        surface,
        context_text,
        candidates,
    )

    if remembered:
        return (
            remembered["selected"],
            "distilled_memory",
            remembered["confidence"],
        )

    choice = teacher.choose(
        context_text,
        candidates,
        decision_type,
        descriptions=descriptions,
    )

    if not choice:
        return None, "teacher_rejected", 0.0

    distilled.learn(
        decision_type,
        surface,
        context_text,
        candidates,
        choice["selected"],
        choice["confidence"],
    )

    return (
        choice["selected"],
        "llm_teacher",
        choice["confidence"],
    )


def apply_live_distillation(
    question,
    parse,
    hypotheses,
    graph,
    distilled,
    teacher,
):
    if not hypotheses:
        return hypotheses, {
            "source": "none",
            "decision": None,
        }

    # Sense decision for resolved concept.
    selected0 = hypotheses[0]

    if (
        selected0.subject
        and (
            selected0.intent
            in {
                "concept_lookup",
                "relation_lookup",
            }
        )
    ):
        senses = candidate_senses(
            graph,
            selected0.subject,
            limit=12,
        )

        if len(senses) > 1:
            sense_names = [
                item["node"]
                for item in senses
            ]
            sense_descriptions = {
                item["node"]:
                    item["definition"]
                for item in senses
            }

            chosen, source, confidence = (
                distilled_choice(
                    teacher,
                    distilled,
                    "sense",
                    selected0.subject,
                    question,
                    sense_names,
                    sense_descriptions,
                )
            )

            if chosen:
                selected_sense = next(
                    (
                        item
                        for item in senses
                        if item["node"] == chosen
                    ),
                    None,
                )

                if selected_sense:
                    # Constrain all later hypotheses to the selected sense
                    # by adding it to evidence. We do not rewrite graph facts.
                    new_hypotheses = [
                        Hypothesis(
                            h.subject,
                            h.relation,
                            h.intent,
                            h.lexical_score,
                            {
                                **h.evidence,
                                "distilled_sense":
                                    selected_sense,
                                "sense_source":
                                    source,
                                "sense_confidence":
                                    confidence,
                                "teacher_format":
                                    "json",
                            },
                        )
                        for h in hypotheses
                    ]

                    return new_hypotheses, {
                        "type": "sense",
                        "selected": chosen,
                        "source": source,
                        "confidence": confidence,
                        "candidates": sense_names,
                    }

    # Relation decision: the LLM chooses which graph relation best matches
    # the language, but only among graph-generated candidates.
    candidate_relations = sorted(
        {
            hypothesis.relation
            for hypothesis in hypotheses
            if hypothesis.relation
        }
    )

    if len(candidate_relations) > 1:
        descriptions = {
            relation:
                relation.replace(
                    "_",
                    " ",
                )
            for relation in candidate_relations
        }

        chosen, source, confidence = (
            distilled_choice(
                teacher,
                distilled,
                "relation",
                selected0.subject or "",
                question,
                candidate_relations,
                descriptions,
            )
        )

        if chosen:
            chosen_hypotheses = [
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.relation
                == chosen
            ]

            if not chosen_hypotheses:
                return hypotheses, {
                    "type": "relation",
                    "selected": chosen,
                    "source": source,
                    "confidence": confidence,
                    "candidates":
                        candidate_relations,
                }

            return chosen_hypotheses, {
                "type": "relation",
                "selected": chosen,
                "source": source,
                "confidence": confidence,
                "candidates":
                    candidate_relations,
            }

    return hypotheses, {
        "type": None,
        "selected": None,
        "source": "none",
        "confidence": 0.0,
    }


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
                "intent_only": True,
                "steps": 0,
                "path": [],
                "target": None,
                "attention": 0,
                "exploration": 0,
                "direct_proof": False,
                "proof_kind": None,
            },
        )

    ranked.sort(
        key=lambda row: (
            -row[0],
            row[1].relation,
            row[1].subject or "",
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
    teacher,
    distilled,
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
    distill_info = apply_live_distillation(
        question,
        parse,
        hypotheses,
        graph,
        distilled,
        teacher,
    )
    distill_seconds = (
        time.perf_counter()
        - t0
    )
    hypotheses = distill_info[0]
    distill_info = distill_info[1]

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

        semantic_match = float(
            hypothesis.lexical_score
        )

        # Proof establishes existence of a route; it must not override a
        # semantically better goal. Direct proof is only a small tie-breaker.
        score = (
            4.0 * semantic_match
            + (
                1.0
                if result.get(
                    "success",
                    False,
                )
                else 0.0
            )
            + (
                0.15
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
        distilled_sense = (
            selected.evidence.get(
                "distilled_sense"
            )
            if isinstance(
                selected.evidence,
                dict,
            )
            else None
        )

        sense_node = (
            distilled_sense.get(
                "node"
            )
            if isinstance(
                distilled_sense,
                dict,
            )
            else None
        )

        definition = graph.definition(
            selected.subject,
            sense_node=sense_node,
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
        "proof_kind": result.get(
            "proof_kind"
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
        "entity_resolution_kind": (
            entity_resolution.get("candidates", [{}])[0].get("kind")
            if entity_resolution.get("candidates")
            else None
        ),
        "route": route,
        "entity_resolution": entity_resolution,
        "goal_selection": [
            {
                "relation": h.relation,
                "intent": h.intent,
                "lexical_score": h.lexical_score,
                "verified": bool(r.get("success", False)),
                "direct_proof": bool(r.get("direct_proof", False)),
                "steps": r.get("steps", 0),
            }
            for _, h, r in ranked[:args.max_hypotheses]
        ],
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
        "distillation": distill_info,
        "timing": {
            "parse_seconds": parse_seconds,
            "hypothesis_seconds": hypothesis_seconds,
            "distillation_seconds": distill_seconds,
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
        "=== V642 FULL SEMANTIC CHAT ===",
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

    distilled = DistilledMemory(
        graph
    )

    teacher = LiveSemanticTeacher(
        args.llm_model,
        temperature=0.05,
    )

    realizer = Realizer(
        args.llm_model
    )

    print(
        "\nTOPICS / EXAMPLES",
        flush=True,
    )

    distilled_stats = distilled.counts()
    print(
        "distilled semantic memory: "
        f"decisions={distilled_stats['decisions']} "
        f"observations={distilled_stats['observations']}",
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
                teacher,
                distilled,
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
                teacher,
                distilled,
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
        "\n=== V642 COMPLETE ===",
        flush=True,
    )
    print(
        f"TRACE  : {trace_path}",
        flush=True,
    )
    distilled_stats = distilled.counts()
    print(
        f"DISTILLED DB: decisions={distilled_stats['decisions']} "
        f"observations={distilled_stats['observations']}",
        flush=True,
    )
    print(
        f"MEMORY : {memory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
