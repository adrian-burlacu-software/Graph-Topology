from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
import sqlite3

from v656_semantic_core import (
    Graph,
    DistilledMemory,
    candidate_senses,
    Context,
    Attention,
    Hypothesis,
    SpaCyParser,
    relation_hypotheses,
    search,
    structural_question_frame,
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


class LocalLLMRuntime:
    """One shared SmolLM tokenizer/model for teacher and realizer."""
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.tokenizer = None
        self.model = None
        self.load_count = 0

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[LLM] loading shared SmolLM3 runtime...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map="auto",
        )
        self.load_count += 1

    def generate(self, prompt, temperature=0.10, max_new_tokens=96):
        self.load()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
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
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            output[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()


class Realizer:
    def __init__(self, runtime):
        self.runtime = runtime

    def load(self):
        self.runtime.load()

    def generate(self, prompt, temperature=0.10, max_new_tokens=96):
        return self.runtime.generate(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ).split("\n",1)[0].strip()

    def grounded_prompt(
        self,
        question,
        hypothesis,
        result,
        definition=None,
    ):
        subject_label = result.get(
            "subject_label",
            hypothesis.subject,
        )
        target_label = result.get(
            "target_label",
            result.get("target"),
        )

        return (
            "Write one short natural answer.\n"
            "The semantic result below is verified.\n"
            "Use only the verified result.\n"
            "Do not output URIs, internal IDs, relation names, "
            "or graph notation unless the user asked for them.\n"
            "Do not introduce another entity.\n\n"
            f"QUESTION: {question}\n"
            f"SUBJECT: {subject_label}\n"
            f"RELATION: {hypothesis.relation}\n"
            f"VERIFIED TARGET: {target_label}\n"
            + (
                f"DEFINITION: {definition}\n"
                if definition
                else ""
            )
            + "ANSWER:"
        )

    def conversation_prompt(self, question, history):
        recent=[]
        for turn in history[-6:]:
            if isinstance(turn, dict):
                recent.append(
                    "USER: "
                    + str(turn.get("question",turn.get("text","")))[:240]
                )
                recent.append(
                    "ASSISTANT: " + str(turn.get("answer",""))[:280]
                )
        return (
            "You are the conversational assistant.\n"
            "Be natural and concise.\n"
            "You may greet the user, tell jokes, and explain general concepts.\n"
            "Do not claim that the semantic graph verified facts it did not verify.\n\n"
            "RECENT:\n"
            + ("\n".join(recent) if recent else "none")
            + f"\nUSER: {question}\nASSISTANT:"
        )




class LiveSemanticTeacher:
    """Candidate-constrained semantic teacher using the shared LLM runtime."""
    def __init__(self, runtime, temperature=0.05):
        self.runtime = runtime
        self.temperature = float(temperature)

    def choose(self, question, candidates, decision_type, descriptions=None, frame=None):
        import json
        import re

        if not candidates:
            return None

        items=[]
        for index,candidate in enumerate(candidates,1):
            item={"id":index,"candidate":str(candidate)}
            if descriptions:
                item["description"]=str(descriptions.get(candidate,""))
            items.append(item)

        if decision_type == "relation_from_frame":
            prompt=(
                "Choose the semantic relation that best expresses what "
                "the user is asking for.\n"
                "Use the structural frame and the supplied relation meanings.\n"
                "Choose exactly one supplied candidate.\n"
                'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
                f"QUESTION: {question}\n"
                f"STRUCTURE: {json.dumps(frame or {}, sort_keys=True, separators=(',', ':'))}\n"
                f"RELATIONS: {json.dumps(items, ensure_ascii=False, separators=(',', ':'))}\n"
                "JSON:"
            )
        else:
            prompt=(
                "Choose ONE graph fact that directly answers the user's question.\n"
                "Use the relation meaning and target together.\n"
                "Do not choose a fact merely because it is associated with the target.\n"
                'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
                f"QUESTION: {question}\n"
                f"FACTS: {json.dumps(items, ensure_ascii=False, separators=(',', ':'))}\n"
                "JSON:"
            )


        raw=self.runtime.generate(
            prompt,
            temperature=self.temperature,
            max_new_tokens=32,
        ).strip()

        match=re.search(
            r"\{.*?\}",
            raw,
            flags=re.DOTALL,
        )
        if not match:
            return None

        try:
            payload=json.loads(match.group(0))
            selected_id=int(payload["selected_id"])
            confidence=float(payload.get("confidence",0.5))
        except Exception:
            return None

        if not 1 <= selected_id <= len(candidates):
            return None

        confidence=max(0.0,min(1.0,confidence))
        return {
            "selected":str(candidates[selected_id-1]),
            "confidence":confidence,
            "raw":raw,
            "selected_id":selected_id,
            "format":"json",
        }



def normalize_graph_fact_candidates(
    candidates,
    subject=None,
):
    normalized = []
    seen = set()

    for item in candidates or []:
        if not isinstance(item, dict):
            continue

        relation = item.get("relation")
        target = item.get(
            "target",
            item.get("object"),
        )

        if not relation or target is None:
            continue

        candidate_subject = item.get(
            "subject",
            subject,
        )

        normalized_item = {
            "id": str(
                item.get(
                    "id",
                    f"{relation}|{target}",
                )
            ),
            "subject": str(
                candidate_subject
                if candidate_subject is not None
                else ""
            ),
            "relation": str(relation),
            "relation_meaning": str(
                item.get(
                    "relation_meaning",
                    str(relation).replace("_", " "),
                )
            ),
            "target": str(target),
            "label": str(
                item.get("label", target)
            ),
            "score": float(
                item.get("score", 0.0) or 0.0
            ),
        }

        key = (
            normalized_item["subject"],
            normalized_item["relation"],
            normalized_item["target"],
        )

        if key in seen:
            continue

        seen.add(key)
        normalized.append(normalized_item)

    return normalized



def choose_relation_from_frame(
    teacher,
    graph,
    parse,
    hypotheses,
):
    candidates = sorted({
        str(h.relation)
        for h in hypotheses
        if h.relation
    })

    if not candidates:
        return None, {
            "source": "none",
            "candidates": [],
        }

    frame = structural_question_frame(parse)
    frame_key = graph.relation_frame_key(frame)

    remembered = graph.relation_frame_lookup(
        frame_key,
        candidates,
    )

    if remembered:
        return remembered["selected"], {
            "source": remembered["source"],
            "confidence": remembered["confidence"],
            "frame_key": frame_key,
            "frame": frame,
            "candidates": candidates,
        }

    descriptions = {
        relation: relation.replace("_", " ")
        for relation in candidates
    }

    choice = teacher.choose(
        parse.text,
        candidates,
        "relation_from_frame",
        descriptions=descriptions,
        frame=frame,
    )

    if not choice:
        return None, {
            "source": "teacher_rejected",
            "confidence": 0.0,
            "frame_key": frame_key,
            "frame": frame,
            "candidates": candidates,
        }

    graph.relation_frame_learn(
        frame_key,
        candidates,
        choice["selected"],
        choice["confidence"],
    )

    return choice["selected"], {
        "source": "llm_teacher",
        "confidence": choice["confidence"],
        "frame_key": frame_key,
        "frame": frame,
        "candidates": candidates,
    }



def choose_graph_fact(
    teacher,
    distilled,
    graph,
    question,
    subject,
    candidates,
):
    candidates = normalize_graph_fact_candidates(
        candidates,
        subject=subject,
    )

    if not candidates:
        return None, {
            "source": "none",
            "selected": None,
            "candidates": [],
        }

    candidate_ids = [
        item["id"]
        for item in candidates
    ]

    descriptions = {
        item["id"]: (
            f"{item['relation']} | "
            f"{item['relation_meaning']} | "
            f"{item['label']} | "
            f"score={item['score']:.2f}"
        )
        for item in candidates
    }


    chosen, source, confidence = distilled_choice(
        teacher,
        distilled,
        "graph_fact_v656",
        subject,
        question,
        candidate_ids,
        descriptions,
    )

    if not chosen:
        return None, {
            "source": source,
            "selected": None,
            "confidence": confidence,
            "candidates": candidate_ids,
        }

    selected = next(
        (
            item
            for item in candidates
            if item["id"] == chosen
        ),
        None,
    )

    return selected, {
        "source": source,
        "selected": chosen,
        "confidence": confidence,
        "candidates": candidate_ids,
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

    selected0 = hypotheses[0]

    # Learn language -> semantic relation from the frozen structural frame.
    # The frame itself contains no semantic relation mapping.
    if selected0.intent in {
        "concept_lookup",
        "relation_lookup",
    }:
        selected_relation, relation_info = (
            choose_relation_from_frame(
                teacher,
                graph,
                parse,
                hypotheses,
            )
        )

        if selected_relation:
            narrowed = [
                Hypothesis(
                    h.subject,
                    h.relation,
                    h.intent,
                    h.lexical_score,
                    {
                        **h.evidence,
                        "frame_signature":
                            relation_info["frame"],
                        "frame_key":
                            relation_info["frame_key"],
                        "relation_from_frame":
                            selected_relation,
                        "relation_frame_source":
                            relation_info["source"],
                        "relation_frame_confidence":
                            relation_info.get(
                                "confidence",
                                0.0,
                            ),
                    },
                )
                for h in hypotheses
                if h.relation == selected_relation
            ]

            if narrowed:
                hypotheses = narrowed

    # Contextual follow-up: when a pronoun resolved the subject, choose among
    # actual observed outgoing facts. This prevents unrelated relations from
    # winning merely because they have a verified edge.
    if (
        selected0.subject
        and selected0.evidence.get(
            "entity_resolution",
            {},
        ).get("status")
        == "context_resolved"
    ):
        candidates = graph.outgoing_candidates(
            selected0.subject,
            question,
            limit=24,
            question_frame="general",
        )

        selected_fact, fact_info = choose_graph_fact(
            teacher,
            distilled,
            graph,
            question,
            selected0.subject,
            candidates,
        )

        fact_info["question_frame"] = (
            "teacher_relation_from_frame"
        )

        if selected_fact:
            matching = [
                Hypothesis(
                    selected0.subject,
                    selected_fact["relation"],
                    "relation_lookup",
                    selected_fact["score"],
                    {
                        **selected0.evidence,
                        "graph_fact_candidate":
                            selected_fact,
                        "selected_fact_target":
                            selected_fact["target"],
                        "fact_source":
                            fact_info["source"],
                        "fact_relation_meaning":
                            selected_fact.get(
                                "relation_meaning",
                                selected_fact["relation"].replace("_"," "),
                            ),
                        "fact_confidence":
                            fact_info["confidence"],
                    },
                )
            ]

            return matching, {
                "type": "graph_fact",
                "selected": selected_fact["id"],
                "source": fact_info["source"],
                "confidence":
                    fact_info["confidence"],
                "candidates": fact_info["candidates"],
            }

    # Sense distillation: ask the teacher to choose only among graph-provided
    # WordNet senses. The selected sense is attached to every candidate
    # hypothesis and becomes authoritative for downstream proof.
    if (
        selected0.subject
        and selected0.intent
        in {
            "concept_lookup",
            "relation_lookup",
        }
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

            chosen, source, confidence = distilled_choice(
                teacher,
                distilled,
                "sense",
                selected0.subject,
                question,
                sense_names,
                sense_descriptions,
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
                    hypotheses = [
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
                                    "json"
                                    if source == "llm_teacher"
                                    else "distilled_memory",
                            },
                        )
                        for h in hypotheses
                    ]

                    return hypotheses, {
                        "type": "sense",
                        "selected": chosen,
                        "source": source,
                        "confidence": confidence,
                        "candidates": sense_names,
                    }

    # Relation distillation: again, constrain the teacher to relations emitted
    # by the graph/controller rather than allowing it to invent one.
    candidate_relations = sorted(
        {
            hypothesis.relation
            for hypothesis in hypotheses
            if hypothesis.relation
        }
    )

    if len(candidate_relations) > 1:
        relation_meanings = {
            "related_to": "general semantic association",
            "is_a": "type or category",
            "has_part": "parts or components",
            "part_of": "belongs to or is a component of",
            "has_property": "property or characteristic",
            "capable_of": "ability or action",
            "at_location": "location or place",
            "antonym": "opposite or contrasting concept",
            "causes": "cause or effect",
            "used_for": "purpose or use",
            "made_of": "material or substance",
        }

        descriptions = {
            relation:
                relation_meanings.get(
                    relation,
                    relation.replace("_", " "),
                )
            for relation in candidate_relations
        }

        chosen, source, confidence = distilled_choice(
            teacher,
            distilled,
            "relation",
            selected0.subject or "",
            question,
            candidate_relations,
            descriptions,
        )

        if chosen:
            chosen_hypotheses = [
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.relation == chosen
            ]

            if chosen_hypotheses:
                chosen_hypotheses = [
                    Hypothesis(
                        h.subject,
                        h.relation,
                        h.intent,
                        h.lexical_score,
                        {
                            **h.evidence,
                            "relation_source": source,
                            "relation_confidence": confidence,
                        },
                    )
                    for h in chosen_hypotheses
                ]

                return chosen_hypotheses, {
                    "type": "relation",
                    "selected": chosen,
                    "source": source,
                    "confidence": confidence,
                    "candidates": candidate_relations,
                }

    return hypotheses, {
        "type": None,
        "selected": None,
        "source": "none",
        "confidence": 0.0,
    }



def validate_grounded_answer(
    answer,
    hypothesis,
    result,
):
    text = str(answer or "").strip()
    if not text:
        return False

    allowed = {
        str(hypothesis.subject or "").rstrip(".,"),
        str(result.get("target") or "").rstrip(".,"),
    }

    entities = re.findall(
        r"https?://\S+|en:[A-Za-z0-9_.:-]+",
        text,
    )

    return all(
        item.rstrip(".,") in allowed
        for item in entities
    )



def clean_surface_answer(answer):
    text=str(answer or "").strip()
    text=text.replace("```","").strip()
    text=re.sub(
        r"^(answer|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text



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
        selected_fact = (
            hypothesis.evidence.get(
                "graph_fact_candidate"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        if (
            selected_fact
            and hypothesis.evidence.get(
                "entity_resolution",
                {},
            ).get("status")
            == "context_resolved"
        ):
            result = graph.prove_edge(
                hypothesis.subject,
                selected_fact["relation"],
                selected_fact["target"],
            )
        else:
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

    realization_cache = None
    realization_guard = "not_applicable"

    if result.get("success", False):
        result["subject_label"] = graph.node_label(
            selected.subject
        )
        if result.get("target") is not None:
            result["target_label"] = (
                graph.node_label(
                    result["target"]
                )
                if (
                    isinstance(
                        result.get("target"),
                        str,
                    )
                    and (
                        result.get("target","").startswith("en:")
                        or result.get("target","").startswith("http")
                    )
                )
                else result.get("target")
            )

    if result.get(
        "success",
        False,
    ):
        realization_cache = graph.get_realized_answer(
            question,
            selected.subject,
            selected.relation,
            result.get("target"),
            result.get("path", []),
        )

        if realization_cache:
            answer = realization_cache["answer"]
            mode = "grounded_cache"
        else:
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
                distilled_sense.get("node")
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

            answer = clean_surface_answer(
                realizer.generate(
                    realizer.grounded_prompt(
                        question,
                        selected,
                        result,
                        definition,
                    ),
                    temperature=0.05,
                )
            )

            if not validate_grounded_answer(
                answer,
                selected,
                result,
            ):
                answer = (
                    f"{selected.subject} "
                    f"{selected.relation.replace('_', ' ')} "
                    f"{result.get('target')}."
                )
                realization_guard = "fallback_exact_fact"
            else:
                realization_guard = "accepted"

            graph.save_realized_answer(
                question,
                selected.subject,
                selected.relation,
                result.get("target"),
                result.get("path", []),
                answer,
            )

            mode = "grounded"

    elif selected.intent == "entity_unresolved":
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
        "selected_fact_target": (
            selected.evidence.get(
                "selected_fact_target"
            )
            if isinstance(
                selected.evidence,
                dict,
            )
            else None
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
        "context": {
            "active_subject_before_turn": memory.active_subject,
            "selected_subject": selected.subject,
        },
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
        "realization_cache": realization_cache,
        "realization_guard": realization_guard,
        "timing": {
            "parse_seconds": parse_seconds,
            "hypothesis_seconds": hypothesis_seconds,
            "distillation_seconds": distill_seconds,
            "search_seconds": search_seconds,
            "llm_seconds": (
                0.0
                if realization_cache
                else llm_seconds
            ),
            "total_seconds": total_seconds,
            "realization_seconds": llm_seconds,
        },
    }

    memory.active_subject = selected.subject
    memory.turns.append(
        trace
    )
    memory.turns = memory.turns[-256:]
    memory.save()

    return answer, trace


def preflight_symbol_audit():
    import builtins
    import ast as _ast
    from pathlib import Path as _Path

    tree = _ast.parse(
        _Path(__file__).read_text(
            encoding="utf-8"
        )
    )

    defined = {
        node.name
        for node in _ast.walk(tree)
        if isinstance(
            node,
            (
                _ast.FunctionDef,
                _ast.AsyncFunctionDef,
                _ast.ClassDef,
            ),
        )
    }

    imported = set()

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
            )
        elif isinstance(node, _ast.ImportFrom):
            imported.update(
                alias.asname or alias.name
                for alias in node.names
            )

    builtins_set = set(
        dir(builtins)
    )

    calls = {
        node.func.id
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Name)
    }

    missing = sorted(
        name
        for name in calls
        if name not in defined
        and name not in imported
        and name not in builtins_set
    )

    # These are deliberately resolved dynamically inside the shared LLM
    # runtime after startup.
    allowed = {
        "torch",
        "transformers",
        "AutoTokenizer",
        "AutoModelForCausalLM",
    }

    missing = [
        name
        for name in missing
        if name not in allowed
    ]

    if missing:
        raise RuntimeError(
            "Gateway symbol audit failed: "
            + ", ".join(missing)
        )

    return True



def main():
    preflight_symbol_audit()
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
        "=== V656 FULL SEMANTIC CHAT ===",
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

    llm_runtime = LocalLLMRuntime(
        args.llm_model
    )

    teacher = LiveSemanticTeacher(
        llm_runtime,
        temperature=0.05,
    )

    realizer = Realizer(
        llm_runtime
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

    realized_stats = graph.realized_answer_stats()
    print(
        "semantic answer cache: "
        f"entries={realized_stats['entries']} "
        f"hits={realized_stats['hits']}",
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
        "\n=== V656 COMPLETE ===",
        flush=True,
    )
    print(
        f"TRACE  : {trace_path}",
        flush=True,
    )
    distilled_stats = distilled.counts()
    realized_stats = graph.realized_answer_stats()
    print(
        f"DISTILLED DB: decisions={distilled_stats['decisions']} "
        f"observations={distilled_stats['observations']}",
        flush=True,
    )
    print(
        f"REALIZED CACHE: entries={realized_stats['entries']} "
        f"hits={realized_stats['hits']}",
        flush=True,
    )
    print(
        f"MEMORY : {memory_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
