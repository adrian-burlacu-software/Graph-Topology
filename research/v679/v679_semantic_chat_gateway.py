from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
import sqlite3
import threading

from v679_semantic_core import (
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
    normalize_question_text,
)
from v679_memory import RamSemanticMemory, SharedCheckpoint, SharedDistilledMemory
from v679_attention import AttentionController, DistilledAttentionPolicy


def append_trace(
    path,
    payload,
    experience_store="",
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
    if experience_store:
        import sys
        v681_dir = Path(__file__).resolve().parents[1] / "v681"
        if str(v681_dir) not in sys.path:
            sys.path.insert(0, str(v681_dir))
        from experience import ExperienceStore, chat_trace_experience
        store = ExperienceStore(experience_store)
        try:
            store.append(chat_trace_experience(payload))
        finally:
            store.close()


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


class WorkerDiscoveryReader:
    """Read derived worker evidence without treating it as source graph truth."""

    def __init__(self, shared_memory):
        self.path = Path(shared_memory)

    def discoveries(self, limit=10):
        if not self.path.exists():
            return []
        try:
            with sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
            ) as connection:
                rows = connection.execute(
                    """
                    SELECT kind,subject,relation,object,feature_json,positive,confidence,
                           derivation_depth,key
                    FROM semantic_knowledge
                    WHERE kind='relation_composition'
                      AND subject IN ('en:animal', 'en:bear', 'en:dog')
                      AND positive > negative
                      AND provenance IN ('derived', 'arbitrated')
                      AND relation='is_a->is_a'
                      AND object != subject
                    ORDER BY subject,object,positive DESC,confidence DESC,key
                    """,
                ).fetchall()
        except sqlite3.Error:
            return []

        by_subject = {"en:animal": [], "en:bear": [], "en:dog": []}
        preferred_targets = (
            "animal", "mammal", "carnivore", "canine", "canid", "organism",
            "living thing", "vertebrate", "quadruped", "predator",
        )
        for row in rows:
            target = str(row[3]).removeprefix("en:")
            if (
                not any(term in target for term in preferred_targets)
                or target in {
                    "animal or body part", "mammal genus", "another word for animal companion",
                    "non person animal",
                }
            ):
                continue
            by_subject[str(row[1])].append(row)
        rows = [
            row
            for subject in ("en:animal", "en:bear", "en:dog")
            for row in sorted(
                by_subject[subject],
                key=lambda row: (
                    next(
                        (
                            index
                            for index, term in enumerate(preferred_targets)
                            if term in str(row[3]).removeprefix("en:")
                        ),
                        len(preferred_targets),
                    ),
                    str(row[3]),
                ),
            )
        ]
        by_subject = {"en:animal": [], "en:bear": [], "en:dog": []}
        seen_questions = set()
        for row in rows:
            discovery = {
                "kind": row[0],
                "subject": row[1],
                "relation": row[2],
                "object": row[3],
                "feature": json.loads(row[4]),
                "positive": int(row[5]),
                "confidence": float(row[6]),
                "derivation_depth": int(row[7]),
                "key": row[8],
            }
            topic = self.topic_for(discovery)
            if topic in seen_questions:
                continue
            seen_questions.add(topic)
            by_subject[discovery["subject"]].append(discovery)
        selected = []
        for index in range(max(len(items) for items in by_subject.values())):
            for subject in ("en:animal", "en:bear", "en:dog"):
                if index < len(by_subject[subject]):
                    selected.append(by_subject[subject][index])
                    if len(selected) >= int(limit):
                        return selected
        return selected

    @staticmethod
    def topic_for(discovery):
        target = discovery["object"].removeprefix("en:")
        article = "an" if target[:1] in "aeiou" else "a"
        return (
            f"Is {discovery['subject'].removeprefix('en:')} "
            f"{article} {target}?"
        )

    def topics(self, limit=10):
        return [
            self.topic_for(discovery)
            for discovery in self.discoveries(limit)
        ]

    def answer(self, question):
        normalized = str(question).strip().lower()
        for discovery in self.discoveries(limit=100):
            if self.topic_for(discovery).lower() == normalized:
                subject_label = discovery["subject"].removeprefix("en:")
                target = discovery["object"].removeprefix("en:")
                article = "an" if target[:1] in "aeiou" else "a"
                return {
                    "answer": (
                        f"Yes, {subject_label} is {article} {target}."
                    ),
                    "previous_relation": discovery["subject"],
                    "next_relation": discovery["object"],
                    "count": discovery["positive"],
                    "confidence": discovery["confidence"],
                    "derivation_depth": discovery["derivation_depth"],
                    "kind": discovery["kind"],
                    "record_key": discovery["key"],
                    "available": True,
                }
        return None


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
            clean_up_tokenization_spaces=False,
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

    def choose_goal(
        self,
        question,
        candidates,
        descriptions=None,
        frame=None,
    ):
        import json
        import re

        items=[
            {
                "id":i,
                "goal":str(candidate),
                "meaning":str(
                    (descriptions or {}).get(
                        candidate,
                        "",
                    )
                ),
            }
            for i,candidate
            in enumerate(candidates,1)
        ]

        prompt=(
            "Choose the USER'S SEMANTIC GOAL.\n"
            "You are not answering the user's question.\n"
            "The candidate goals are the complete allowed vocabulary.\n"
            "Do not mention or infer database relation names.\n"
            "Use the grammatical structure and question wording.\n"
            "For 'what is X?', choose definition.\n"
            "For 'is X Y?' where Y is a characteristic, choose property.\n"
            "For 'is X a Y?' choose type.\n"
            "For parts/components choose part.\n"
            "For abilities/actions choose capability.\n"
            "For place/location choose location.\n"
            "Choose the goal that represents what information is requested, "
            "not a goal that merely has a graph edge to a matching word.\n"
            'Return only JSON: {"selected_id": 1, "confidence": 0.0}\n\n'
            f"QUESTION: {question}\n"
            f"FRAME: {json.dumps(frame or {},ensure_ascii=False,sort_keys=True,separators=(',',':'))}\n"
            f"GOALS: {json.dumps(items,ensure_ascii=False,separators=(',',':'))}\n"
            "JSON:"
        )

        raw=self.runtime.generate(
            prompt,
            temperature=min(
                self.temperature,
                0.05,
            ),
        )

        match=re.search(
            r"\{.*\}",
            str(raw),
            re.DOTALL,
        )
        if not match:
            return None

        try:
            data=json.loads(match.group(0))
            selected_id=int(data.get("selected_id",0))
            confidence=float(data.get("confidence",0.0))
        except Exception:
            return None

        if not 1 <= selected_id <= len(items):
            return None

        return {
            "selected":items[
                selected_id-1
            ]["goal"],
            "confidence":max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
        }


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

        if decision_type == "relation_from_frame_v662":
            prompt=(
                "Interpret the user's question as a semantic relation query.\n"
                "Do NOT answer the user.\n"
                "Do NOT choose a relation only because its example contains "
                "the same words.\n"
                "Use the grammatical argument structure and the meaning "
                "demonstrated by the graph examples.\n"
                "A relation is appropriate only if its examples express the "
                "same kind of relationship between the subject and the "
                "requested property, type, part, action, place, or other "
                "semantic target.\n"
                "Choose exactly one supplied relation.\n"
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
            confidence=float(payload.get("confidence",0.0))
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



def question_argument_frame(parse):
    tokens=[
        item
        for item in (parse.tokens or [])
        if isinstance(item,dict)
    ]
    predicates=[
        str(item.get("text",""))
        for item in tokens
        if item.get("dep") in {
            "acomp",
            "attr",
            "dobj",
            "obj",
            "pobj",
            "ROOT",
        }
        and item.get("text")
    ]

    predicate_lemmas=[
        str(item.get("lemma",""))
        for item in tokens
        if item.get("dep") in {
            "acomp",
            "attr",
            "dobj",
            "obj",
            "pobj",
        }
        and item.get("lemma")
    ]

    return {
        "question":str(
            parse.question or ""
        ),
        "root":str(
            parse.root_lemma or ""
        ),
        "subjects":[
            str(x)
            for x in (parse.subjects or [])
        ],
        "objects":[
            str(x)
            for x in (parse.objects or [])
        ],
        "predicate_tokens":predicates,
        "predicate_lemmas":predicate_lemmas,
        "pronoun_subject":any(
            item.get("pos")=="PRON"
            and item.get("dep")=="nsubj"
            for item in tokens
        ),
    }


def classify_turn_mode(parse):
    """
    Classify the speech act before semantic relation inference.

    Semantic grounding is for explicit information-seeking questions.
    Ordinary conversation/request utterances (greetings, imperatives,
    social requests such as "tell me a joke") bypass the graph entirely.

    This is structural rather than concept/word based: interrogative/WH
    structure routes to semantic lookup; non-interrogative utterances route
    to conversation. Embedded WH clauses remain eligible for grounding.
    """
    tokens=[item for item in (parse.tokens or []) if isinstance(item,dict)]
    text=str(parse.text or "").strip()

    wh_questions={
        "WH_WHAT","WH_WHO","WH_WHERE","WH_WHEN",
        "WH_WHY","WH_HOW","WH_WHICH",
    }
    question_kind=str(getattr(parse,"question","") or "")
    has_wh=any(
        str(item.get("pos","")).upper()=="PRON"
        and str(item.get("text","")).lower() in
        {"what","who","where","when","why","how","which"}
        for item in tokens
    )
    has_question_mark=text.endswith("?")

    if (
        question_kind in wh_questions
        or has_wh
        or has_question_mark
        or re.fullmatch(
            r"\s*(?:the\s+)?(?:part|parts|component|components)\s+of\s+.+\s*",
            text.lower(),
        )
    ):
        return "semantic"

    # Imperatives and ordinary non-interrogative utterances are conversational
    # requests/statements unless they contain an embedded WH question above.
    return "conversation"


def question_target_terms(parse):
    """
    Extract explicit non-subject semantic arguments from the frozen parse.

    This is structural, not relation-specific: nouns/adjectives/objects from
    the question become constraints on which graph object may satisfy the
    already-selected semantic goal.
    """
    tokens=[
        item
        for item in (parse.tokens or [])
        if isinstance(item,dict)
    ]
    copular_match = re.fullmatch(
        r"is\s+(?:(?:a|an|the)\s+)?(.+?)\s+"
        r"(?:(?:a|an|the)\s+)?(.+?)\??",
        normalize_question_text(parse.text),
    )
    if copular_match:
        target = copular_match.group(2).strip()
        if target:
            return [target]
    subject_indexes = {
        index
        for index, item in enumerate(tokens)
        if str(item.get("dep", "")).lower() in {"nsubj", "nsubjpass"}
    }
    auxiliary_indexes = [
        index
        for index, item in enumerate(tokens)
        if str(item.get("lemma") or item.get("text") or "").lower()
        in {"do", "does", "did", "can", "could", "be", "is", "are", "was", "were"}
    ]
    if (
        not subject_indexes
        and str(parse.question or "") in {"QUESTION", "WH_WHAT", "WH_WHICH"}
    ):
        for auxiliary_index in auxiliary_indexes:
            structural_subject = next(
                (
                    index
                    for index, item in enumerate(
                        tokens[auxiliary_index + 1:],
                        auxiliary_index + 1,
                    )
                    if str(item.get("pos", "")).upper() in {"NOUN", "PROPN"}
                ),
                None,
            )
            if structural_subject is not None:
                subject_indexes = {structural_subject}
                break
    if (
        not subject_indexes
        and str(parse.question or "") == "QUESTION"
        and tokens
        and str(tokens[0].get("pos", "")).upper() == "AUX"
    ):
        copular_nouns = [
            index
            for index, item in enumerate(tokens[1:], 1)
            if str(item.get("pos", "")).upper() in {"NOUN", "PROPN"}
        ]
        subject_indexes = set(copular_nouns[:1])

    stop={
        "a","an","the","it","he","she","they","them",
        "this","that","these","those","what","who","where",
        "when","why","how","which","is","are","was","were",
        "be","have","has","can","could","do","does","did","of","to",
        "for","with","and","or","on","in","at","from",
    }

    terms=[]
    for index, item in enumerate(tokens):
        pos=str(item.get("pos","")).upper()
        dep=str(item.get("dep","")).lower()
        forms=[
            str(item.get("text") or "").lower().strip(),
            str(item.get("lemma") or "").lower().strip(),
        ]

        if not any(form and form not in stop for form in forms):
            continue
        if pos not in {"NOUN","PROPN","ADJ","ADV","NUM","VERB"}:
            continue
        if (
            index in subject_indexes
            or dep in {"nsubj","nsubjpass","det","aux","auxpass"}
        ):
            continue
        for form in forms:
            if form and form not in stop and form not in terms:
                terms.append(form)

    phrase_terms = []
    subject_surfaces = {
        str(tokens[index].get("text") or "").lower()
        for index in subject_indexes
    }
    for chunk in (parse.noun_chunks or []):
        words = re.findall(r"[a-z0-9]+", str(chunk).lower())
        phrase = " ".join(
            word
            for word in words
            if word not in stop | subject_surfaces
        )
        if (
            phrase
            and phrase not in subject_surfaces
            and phrase not in phrase_terms
        ):
            phrase_terms.append(phrase)

    if subject_indexes:
        predicate_words = [
            str(item.get("text") or "").lower()
            for item in tokens[max(subject_indexes) + 1:]
            if str(item.get("pos", "")).upper() != "PUNCT"
        ]
        while predicate_words and predicate_words[0] in {
            "is", "are", "was", "were", "be", "have", "has",
            "can", "could", "do", "does", "did", "a", "an", "the",
        }:
            predicate_words.pop(0)
        predicate_phrase = " ".join(predicate_words)
        if predicate_phrase and predicate_phrase not in phrase_terms:
            phrase_terms.append(predicate_phrase)

    # Keep the explicit object/predicate ordering visible for the trace.
    if any(
        word in {"part", "parts", "component", "components"}
        for word in re.findall(r"[a-z]+", parse.text.lower())
    ):
        terms = [
            term for term in terms
            if term not in {"part", "parts", "component", "components"}
        ]
        phrase_terms = [
            term for term in phrase_terms
            if term not in {"part", "parts", "component", "components"}
        ]
        if requested_part_list(parse.text):
            return []
    return (phrase_terms or terms)[:8]


def is_definition_form(parse):
    question_text = normalize_question_text(parse.text)
    words = re.findall(r"[a-z]+", question_text)
    return (
        len(words) >= 2
        and words[0] in {"what", "who"}
        and words[1] in {"is", "are", "was", "were"}
        and not re.search(
            r"\b(?:part|parts|component|components)\b",
            question_text,
        )
    )


def restrict_definition_hypotheses(parse, hypotheses):
    if is_definition_form(parse):
        return hypotheses
    return [
        hypothesis for hypothesis in hypotheses
        if hypothesis.relation not in {"definition", "has_sense"}
    ]


def choose_semantic_goal(
    teacher,
    graph,
    parse,
    subject,
    memory=None,
):
    goals=[
        item
        for item in graph.semantic_goal_candidates()
        if item.get("available_relations")
    ]

    if not goals:
        return None,{
            "source":"no_available_goal",
            "candidates":[],
        }

    target_terms = question_target_terms(parse)
    available_goals = {item["goal"] for item in goals}
    normalized_question = normalize_question_text(parse.text)
    copular_match = re.fullmatch(
        r"is\s+(?:(?:a|an|the)\s+)?(.+?)\s+"
        r"(?:(?:a|an|the)\s+)?(.+?)\??",
        normalized_question,
    )
    if copular_match and subject:
        copular_target = copular_match.group(2).strip()
        if (
            "type" in available_goals
            and graph.has_goal_path(
                subject,
                "type",
                [copular_target],
                max_depth=2,
            )
        ):
            return "type", {
                "source": "structural_copular_type_path",
                "confidence": 1.0,
                "candidates": [item["goal"] for item in goals],
                "candidate_details": goals,
                "frame": question_argument_frame(parse),
                "syntax_frame": structural_question_frame(parse),
            }
        for goal in ("type", "property"):
            if goal not in available_goals:
                continue
            if graph.find_goal_facts(
                subject,
                goal,
                query_text=parse.text,
                target_terms=[copular_target],
                limit=1,
            ):
                return goal, {
                    "source": "structural_copular_graph_evidence",
                    "confidence": 1.0,
                    "candidates": [item["goal"] for item in goals],
                    "candidate_details": goals,
                    "frame": question_argument_frame(parse),
                    "syntax_frame": structural_question_frame(parse),
                }
    if requested_part_list(parse.text) and "part" in available_goals:
        return "part", {
            "source": "structural_part_inventory",
            "confidence": 1.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }
    target_words = {
        word
        for term in target_terms
        for word in re.findall(r"[a-z0-9]+", str(term).lower())
    }
    target_pos = {
        str(item.get("pos", "")).upper()
        for item in (parse.tokens or [])
        if str(item.get("text") or item.get("lemma") or "").lower() in target_words
    }
    if (
        str(parse.root_lemma or "").lower() == "be"
        and target_terms
        and "type" in available_goals
        and target_pos.intersection({"NOUN", "PROPN"})
    ):
        return "type", {
            "source": "structural_copular_type",
            "confidence": 1.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }
    if (
        str(parse.root_lemma or "").lower() == "be"
        and target_terms
        and "property" in available_goals
        and "ADJ" in target_pos
    ):
        return "property", {
            "source": "structural_copular_property",
            "confidence": 1.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }
    if (
        str(parse.root_lemma or "").lower() in {"have", "has"}
        and subject
        and target_terms
        and "part" in available_goals
    ):
        return "part", {
            "source": "structural_possessive_part",
            "confidence": 1.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }
    direct_goal_matches = {
        item["goal"]: graph.find_goal_facts(
            subject,
            item["goal"],
            query_text=parse.text,
            target_terms=target_terms,
            limit=2,
        )
        for item in goals
    } if subject and target_terms else {}
    matching_goals = [
        goal for goal, facts in direct_goal_matches.items()
        if facts
    ]
    specific_matching_goals = [
        goal for goal in matching_goals
        if goal != "association"
    ]
    if specific_matching_goals:
        matching_goals = specific_matching_goals
    if len(matching_goals) == 1:
        return matching_goals[0], {
            "source": "direct_argument_evidence",
            "confidence": 1.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }

    # Explicit arguments constrain the graph fact, not merely answer wording.
    # Without a matching fact, selecting another relation would turn a failed
    # breed/part/property request into an unrelated but provable answer.
    if target_terms and not is_definition_form(parse):
        return None, {
            "source": "no_direct_argument_evidence",
            "confidence": 0.0,
            "candidates": [item["goal"] for item in goals],
            "candidate_details": goals,
            "frame": question_argument_frame(parse),
            "syntax_frame": structural_question_frame(parse),
        }

    # Definition is available only to grammatical definition requests. Other
    # questions must not return a definition merely because it is easy to prove.
    if not is_definition_form(parse):
        goals = [
            item for item in goals
            if item["goal"] != "definition"
        ]

    descriptions={
        item["goal"]:item["meaning"]
        for item in goals
    }

    syntax_frame=structural_question_frame(parse)
    argument_frame=question_argument_frame(parse)
    frame_key=graph.relation_frame_key(structural_question_frame(parse))

    choice=teacher.choose_goal(
        parse.text,
        [item["goal"] for item in goals],
        descriptions=descriptions,
        frame={
            "syntax":syntax_frame,
            "arguments":argument_frame,
        },
    )

    if not choice:
        return None,{
            "source":"teacher_rejected",
            "candidates":[item["goal"] for item in goals],
            "candidate_details":goals,
        }

    selected=str(
        choice.get("selected") or ""
    )
    valid={
        item["goal"]
        for item in goals
    }

    if selected not in valid:
        return None,{
            "source":"invalid_teacher_selection",
            "selected":selected,
            "candidates":[item["goal"] for item in goals],
            "candidate_details":goals,
        }

    if memory is not None and hasattr(memory, "goal_learn"):
        memory.goal_learn(
            frame_key,
            [item["goal"] for item in goals],
            selected,
            choice.get("confidence", 0.0),
        )

    return selected,{
        "source":"llm_teacher",
        "confidence":choice.get("confidence",0.0),
        "candidates":[item["goal"] for item in goals],
        "candidate_details":goals,
        "frame":argument_frame,
        "syntax_frame":syntax_frame,
    }



def choose_relation_from_frame(
    teacher,
    graph,
    parse,
    hypotheses,
    relation_limit=32,
):
    subject=(
        hypotheses[0].subject
        if hypotheses
        else None
    )

    if not subject:
        return None,{
            "source":"none",
            "candidates":[],
        }

    evidence_candidates=graph.semantic_relation_candidates(
        subject=subject,
        query_text=parse.text,
        local_relations=[
            str(h.relation)
            for h in hypotheses
            if h.relation
        ],
        limit=relation_limit,
        examples_per_relation=4,
    )

    if not evidence_candidates:
        return None,{
            "source":"none",
            "candidates":[],
        }

    relation_names=[
        item["relation"]
        for item in evidence_candidates
    ]

    frame=structural_question_frame(parse)
    argument_frame=question_argument_frame(parse)
    frame_key=graph.relation_frame_key(
        frame
    )

    remembered=graph.relation_frame_lookup(
        frame_key,
        relation_names,
        min_confidence=0.85,
        min_count=2,
    )

    if remembered:
        return remembered["selected"],{
            "source":remembered["source"],
            "confidence":remembered["confidence"],
            "count":remembered["count"],
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    descriptions={
        item["relation"]:json.dumps(
            {
                "relation":item["relation"],
                "direct_question_matches":
                    item.get(
                        "direct_question_matches",
                        0,
                    ),
                "examples":item.get(
                    "examples",
                    [],
                ),
            },
            ensure_ascii=False,
            separators=(",",":"),
        )
        for item in evidence_candidates
    }

    choice=teacher.choose(
        parse.text,
        relation_names,
        "relation_from_frame_v662",
        descriptions=descriptions,
        frame={
            **frame,
            "argument_frame":
                argument_frame,
        },
    )

    if not choice:
        return None,{
            "source":"teacher_rejected",
            "confidence":0.0,
            "count":0,
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    selected=str(
        choice.get("selected") or ""
    )

    if selected not in relation_names:
        return None,{
            "source":"invalid_teacher_selection",
            "selected":selected,
            "confidence":0.0,
            "count":0,
            "frame_key":frame_key,
            "frame":frame,
            "argument_frame":argument_frame,
            "candidates":relation_names,
            "candidate_details":evidence_candidates,
        }

    graph.relation_frame_learn(
        frame_key,
        relation_names,
        selected,
        choice.get("confidence",0.0),
    )

    return selected,{
        "source":"llm_teacher",
        "confidence":choice.get("confidence",0.0),
        "count":1,
        "frame_key":frame_key,
        "frame":frame,
        "argument_frame":argument_frame,
        "candidates":relation_names,
        "candidate_details":evidence_candidates,
    }









def choose_graph_fact(
    teacher,
    distilled,
    graph,
    question,
    subject,
    candidates,
):
    candidates=normalize_graph_fact_candidates(
        candidates,
        subject=subject,
    )

    if not candidates:
        return None,{
            "source":"none",
            "selected":None,
            "confidence":0.0,
            "candidates":[],
        }

    ordered=sorted(
        candidates,
        key=lambda item:(
            -float(
                item.get(
                    "score",
                    0.0,
                ) or 0.0
            ),
            item["target"].lower(),
            item["relation"],
        ),
    )

    chosen=ordered[0]

    return chosen,{
        "source":"graph_deterministic",
        "selected":chosen["id"],
        "confidence":1.0,
        "candidates":[
            item["id"]
            for item in candidates
        ],
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
        return hypotheses,{
            "source":"none",
            "decision":None,
        }

    base=hypotheses[0]

    if base.intent not in {
        "concept_lookup",
        "relation_lookup",
    }:
        return hypotheses,{
            "source":"none",
            "decision":None,
        }

    goal,info=choose_semantic_goal(
        teacher,
        graph,
        parse,
        base.subject,
        memory=distilled,
    )

    if not goal:
        target_terms = question_target_terms(parse)
        if (
            info.get("source") == "no_direct_argument_evidence"
            and base.subject
        ):
            return [
                Hypothesis(
                    base.subject,
                    "",
                    "relation_lookup",
                    base.lexical_score,
                    {
                        **base.evidence,
                        "target_terms": target_terms,
                        "argument_unverified": True,
                    },
                )
            ], {
                "source": info["source"],
                "decision": None,
            }
        return hypotheses,{
            "source":info.get(
                "source",
                "teacher_rejected",
            ),
            "decision":None,
        }

    evidence={
        **base.evidence,
        "semantic_goal":goal,
        "semantic_goal_source":info[
            "source"
        ],
        "semantic_goal_confidence":info.get(
            "confidence",
            0.0,
        ),
        "semantic_goal_candidates":info[
            "candidates"
        ],
        "semantic_goal_details":info.get(
            "candidate_details",
            [],
        ),
        "argument_frame":info.get(
            "frame",
            {},
        ),
        "syntax_frame":info.get(
            "syntax_frame",
            {},
        ),
        "target_terms":question_target_terms(parse),
    }

    # Keep the clean goal as the hypothesis relation. Raw graph relations are
    # deliberately not exposed to the teacher.
    hypotheses=[
        Hypothesis(
            base.subject,
            goal,
            "relation_lookup",
            base.lexical_score,
            evidence,
        )
    ]

    # Entity resolution + goal-specific fact lookup. This adapter is entirely
    # internal to the graph and cannot be selected by the language model.
    if (
        base.subject
        and base.evidence.get(
            "entity_resolution",
            {},
        ).get("status")
        in {"context_resolved", "resolved"}
    ):
        target_terms=question_target_terms(parse)
        facts=graph.find_goal_facts(
            base.subject,
            goal,
            query_text=question,
            target_terms=target_terms,
            limit=24,
        )

        evidence["target_terms"]=target_terms

        if facts:
            chosen=sorted(
                facts,
                key=lambda item:(
                    item["relation"],
                    item["label"].lower(),
                ),
            )[0]

            hypotheses=[
                Hypothesis(
                    base.subject,
                    goal,
                    "relation_lookup",
                    base.lexical_score,
                    {
                        **evidence,
                        "raw_relation":
                            chosen["relation"],
                        "selected_fact_target":
                            chosen["object"],
                        "selected_fact_label":
                            chosen["label"],
                        "fact_source":
                            "semantic_goal_adapter",
                        "fact_confidence":1.0,
                        "fact_target_terms":question_target_terms(parse),
                    },
                )
            ]

    return hypotheses,{
        "source":info["source"],
        "decision":goal,
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


def requested_part_list(question):
    text = str(question).lower()
    words = set(re.findall(r"[a-z]+", text))
    return bool(
        words.intersection({"part", "parts", "component", "components"})
    ) and (
        bool(words.intersection({"what", "which"}))
        or bool(re.fullmatch(
            r"\s*(?:the\s+)?(?:part|parts|component|components)\s+of\s+.+\s*",
            text,
        ))
    )


def is_polar_question(parse):
    return str(parse.question or "") == "QUESTION"


def verified_polar_answer(selected, result):
    goal = result.get("semantic_goal", selected.relation)
    return (
        "Yes, the semantic graph verifies the requested "
        f"{goal} for {result['subject_label']}: "
        f"{result['target_label']}."
    )


def dispatch_worker_query(checkpoint, question):
    if checkpoint is None:
        return None
    text = normalize_question_text(question)
    match = re.search(r"\b(animal|bear|dog)s?\b", text)
    if not match:
        return None
    try:
        return checkpoint.enqueue_query(question, f"en:{match.group(1)}")
    except sqlite3.Error:
        return None


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
    worker_discoveries=None,
    worker_pool=None,
    controller=None,
):
    started = time.perf_counter()
    controller = controller or AttentionController()

    t0 = time.perf_counter()
    parse = parser.parse(
        question
    )
    parse_seconds = (
        time.perf_counter()
        - t0
    )
    worker_query_id = dispatch_worker_query(worker_pool, question)
    # Worker records support later answers and diagnostics, but never replace
    # graph-authoritative routing for an ordinary user question.
    discovery = None
    if discovery:
        total_seconds = time.perf_counter() - started
        result = {
            "success": bool(discovery.get("available", True)),
            "steps": 0,
            "path": [],
            "target": discovery["next_relation"],
            "attention": 0,
            "exploration": 0,
            "direct_proof": False,
            "proof_kind": "worker_derived_transition",
        }
        trace = {
            "question": question,
            "answer": discovery["answer"],
            "route": {
                "intent": "worker_discovery",
                "subject": discovery["previous_relation"],
                "relation": discovery["next_relation"],
                "mode": "worker_derived",
                **result,
            },
            "parse": asdict(parse),
            "worker_discovery": discovery,
            "worker_pool": {"query_id": worker_query_id},
            "timing": {
                "parse_seconds": parse_seconds,
                "hypothesis_seconds": 0.0,
                "distillation_seconds": 0.0,
                "search_seconds": 0.0,
                "llm_seconds": 0.0,
                "total_seconds": total_seconds,
                "realization_seconds": 0.0,
            },
        }
        memory.turns.append(trace)
        memory.turns = memory.turns[-256:]
        memory.save()
        return discovery["answer"], trace

    # Speech-act routing happens before graph hypothesis generation. This
    # prevents conversational requests such as "tell me a joke" from being
    # interpreted as semantic definitions merely because a concept named
    # "joke" exists in the graph.
    turn_mode = classify_turn_mode(parse)
    if turn_mode == "conversation":
        t_conv = time.perf_counter()
        answer = realizer.generate(
            realizer.conversation_prompt(
                question,
                memory.turns,
            ),
            temperature=0.15,
        )
        llm_seconds = time.perf_counter() - t_conv
        total_seconds = time.perf_counter() - started
        result = {
            "success": False,
            "intent_only": True,
            "steps": 0,
            "path": [],
            "target": None,
            "attention": 0,
            "exploration": 0,
            "direct_proof": False,
            "proof_kind": "conversation",
        }
        trace = {
            "question": question,
            "answer": answer,
            "worker_pool": {"query_id": worker_query_id},
            "route": {
                "intent": "conversation",
                "subject": None,
                "relation": None,
                "mode": "conversation",
                "success": False,
                **result,
            },
            "parse": {
                "text": parse.text,
                "question": parse.question,
                "root": parse.root,
                "root_lemma": parse.root_lemma,
                "subjects": list(parse.subjects or []),
                "objects": list(parse.objects or []),
            },
            "goal_selection": [],
            "hypotheses": [],
            "search": result,
            "distillation": {
                "source": "none",
                "decision": None,
            },
            "realization_cache": None,
            "realization_guard": "not_applicable",
            "timing": {
                "parse_seconds": parse_seconds,
                "hypothesis_seconds": 0.0,
                "distillation_seconds": 0.0,
                "search_seconds": 0.0,
                "llm_seconds": llm_seconds,
                "total_seconds": total_seconds,
                "realization_seconds": llm_seconds,
            },
        }
        memory.turns.append({"question": question, "answer": answer})
        memory.save()
        return answer, trace

    t0 = time.perf_counter()
    hypotheses = relation_hypotheses(
        parse,
        graph,
        memory,
        max_n=args.max_hypotheses,
    )
    hypotheses = restrict_definition_hypotheses(parse, hypotheses)
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

    runtime_hypotheses = []

    for hypothesis in hypotheses:
        semantic_goal = (
            hypothesis.evidence.get(
                "semantic_goal"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        if semantic_goal:
            target_terms = list(hypothesis.evidence.get("target_terms", []) or [])
            # When the user requested an explicit target, do NOT run unconstrained
            # semantic-path search. The adapter must first find a fact whose target
            # is that requested argument, and proof must validate exactly that fact.
            if target_terms and hypothesis.evidence.get("selected_fact_target"):
                runtime_hypotheses.append(hypothesis)
            else:
                for raw_relation in graph.semantic_relations_for_goal(semantic_goal):
                    runtime_hypotheses.append(
                        Hypothesis(
                            hypothesis.subject,
                            raw_relation,
                            hypothesis.intent,
                            hypothesis.lexical_score,
                            {**hypothesis.evidence,
                             "semantic_goal": semantic_goal,
                             "semantic_goal_runtime_relation": raw_relation},
                        )
                    )
        else:
            runtime_hypotheses.append(hypothesis)

    runtime_hypotheses = controller.prioritize_hypotheses(runtime_hypotheses)
    controller.begin_turn(
        runtime_hypotheses[0].subject if runtime_hypotheses else None
    )

    for index,hypothesis in enumerate(runtime_hypotheses):
        controller.begin_hypothesis(hypothesis)
        selected_fact = (
            hypothesis.evidence.get(
                "selected_fact_target"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )
        raw_relation = (
            hypothesis.evidence.get(
                "raw_relation"
            )
            if isinstance(
                hypothesis.evidence,
                dict,
            )
            else None
        )

        if (
            selected_fact
            and raw_relation
            and hypothesis.evidence.get(
                "entity_resolution",
                {},
            ).get("status")
            in {"context_resolved", "resolved"}
        ):
            result=graph.prove_edge(
                hypothesis.subject,
                raw_relation,
                selected_fact,
            )
            if result.get("success"):
                controller.record_direct_proof(hypothesis, result.get("target"))
        else:
            result=search(
                graph,
                attention,
                hypothesis,
                budget=args.goal_budget,
                per_node=args.per_node,
                max_depth=args.max_depth,
                controller=controller,
            )
        controller.record_selected_path(hypothesis, result)

        semantic_match=float(
            hypothesis.lexical_score
        )

        score=(
            4.0*semantic_match
            +(
                1.0
                if result.get(
                    "success",
                    False,
                )
                else 0.0
            )
            +(
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

    semantic_decision = controller.arbitrate(ranked)
    selected_index = semantic_decision["selected_candidate_index"]
    if selected_index is None:
        selected, result = choose_best(ranked)
    else:
        _, selected, result = ranked[selected_index]

    # A plural parts/components question requests the complete direct graph
    # inventory, not a single representative fact or a multi-hop substitute.
    if requested_part_list(question) and selected.subject:
        parts = graph.find_goal_facts(
            selected.subject,
            "part",
            query_text=question,
            target_terms=[],
            limit=100,
        )
        labels = sorted({
            str(item["label"])
            for item in parts
        }, key=str.lower)
        if labels:
            selected.evidence["all_part_labels"] = labels
            result = {
                "success": True,
                "steps": 1,
                "path": ["has_a", "has_part"],
                "target": parts[0]["object"],
                "attention": 0,
                "exploration": 0,
                "direct_proof": True,
                "proof_kind": "direct_part_inventory",
            }

    # Keep the public semantic goal stable even though graph search uses
    # internal raw relations.
    if (
        isinstance(selected.evidence, dict)
        and selected.evidence.get(
            "semantic_goal"
        )
    ):
        result["semantic_goal"] = selected.evidence[
            "semantic_goal"
        ]
        result["semantic_relation"] = selected.evidence[
            "semantic_goal"
        ]

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
        part_labels = (
            selected.evidence.get("all_part_labels")
            if isinstance(selected.evidence, dict)
            else None
        )
        if part_labels:
            answer = (
                f"{result['subject_label']} has: "
                + ", ".join(part_labels)
                + "."
            )
            mode = "grounded"
            realization_cache = {
                "source": "direct_part_inventory",
                "count": len(part_labels),
            }
            realization_guard = "deterministic_part_inventory"
        elif is_polar_question(parse):
            answer = verified_polar_answer(selected, result)
            mode = "grounded"
            realization_guard = "deterministic_polar_fact"
        realization_cache = (
            None
            if selected.relation == "definition" or is_polar_question(parse)
            else graph.get_realized_answer(
            question,
            selected.subject,
            selected.evidence.get(
                "semantic_goal",
                selected.relation,
            )
            if isinstance(
                selected.evidence,
                dict,
            )
            else selected.relation,
            result.get("target"),
            result.get("path", []),
        ))

        if part_labels or is_polar_question(parse):
            pass
        elif realization_cache:
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

            if selected.relation == "definition" and definition:
                answer = f"{result['subject_label']} is {definition}."
                realization_guard = "deterministic_definition"
            else:
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

            if realization_guard != "deterministic_definition" and not validate_grounded_answer(
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

    elif selected.intent == "relation_lookup":
        # A semantic/factual request is never allowed to fall through to
        # conversational generation merely because graph proof failed.
        # Doing so would let the LLM answer from pretrained world knowledge
        # and violate the graph-authoritative grounding boundary.
        goal = (
            selected.evidence.get("semantic_goal")
            if isinstance(selected.evidence, dict)
            else None
        )
        target_terms = (
            selected.evidence.get("target_terms", [])
            if isinstance(selected.evidence, dict)
            else []
        )
        if semantic_decision["outcome"] == "abstain":
            answer = (
                "I don't know: no candidate has verified graph evidence "
                "for that request."
            )
        elif goal and target_terms:
            target_text = ", ".join(str(x) for x in target_terms)
            answer = (
                "I couldn't verify that "
                f"the subject has the requested {goal} ({target_text}) "
                "in the semantic graph."
            )
        else:
            answer = (
                "I couldn't verify that requested fact "
                "in the semantic graph."
            )
        mode = "unverified"

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
        "candidate_evidence": semantic_decision["candidates"],
        "semantic_decision": semantic_decision,
        "attention_controller": controller.trace(),
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



def run_chat_worker(args):
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
        "--shared-memory",
        required=False,
        default="./results/v679_shared_memory.sqlite",
    )
    ap.add_argument(
        "--worker-id",
        type=int,
        default=19,
    )
    ap.add_argument(
        "--total-workers",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--checkpoint-seconds",
        type=int,
        choices=(60,300),
        default=300,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=63300,
    )

    # args is supplied by the V679 runtime

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
        "=== V679 CHAT WORKER ===",
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
    print(
        f"shared   : {Path(args.shared_memory).resolve()} interval={args.checkpoint_seconds}s",
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
    policy_path = str(getattr(args, "attention_policy", "") or "")
    attention_controller = AttentionController(
        policy=DistilledAttentionPolicy.load(policy_path) if policy_path else None
    )

    ram_memory = RamSemanticMemory(worker_id=getattr(args, "worker_id", 19))
    checkpoint = SharedCheckpoint(
        args.shared_memory,
        getattr(args, "worker_id", 19),
        getattr(args, "total_workers", 20),
        args.checkpoint_seconds,
    )
    try:
        checkpoint.sync(ram_memory, "chat", __import__("os").getpid(), force=True)
    except Exception:
        pass
    distilled = SharedDistilledMemory(graph, ram_memory, checkpoint)
    worker_discoveries = WorkerDiscoveryReader(args.shared_memory)

    # Keep online semantic memory synchronized even while the user is idle.
    # The checkpoint itself decides whether the current-time modulus slot belongs
    # to this worker, so polling does not cause writes outside the worker's slot.
    sync_stop = threading.Event()
    def _sync_loop():
        while not sync_stop.wait(0.50):
            try:
                if checkpoint.should_sync():
                    checkpoint.sync(ram_memory, "chat", __import__("os").getpid())
            except Exception as exc:
                try:
                    checkpoint.record_event("error", {"stage":"chat_background_checkpoint", "error":repr(exc)})
                except Exception:
                    pass
    sync_thread = threading.Thread(target=_sync_loop, name="v679-chat-checkpoint", daemon=True)
    sync_thread.start()

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
        f"decisions={distilled_stats.get('decisions', 0)} "
        f"observations={distilled_stats.get('observations', distilled_stats.get('durable_observations', 0))}",
        flush=True,
    )

    realized_stats = graph.realized_answer_stats()
    print(
        "semantic answer cache: "
        f"entries={realized_stats['entries']} "
        f"hits={realized_stats['hits']}",
        flush=True,
    )

    base_topics = [
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

    def current_topics():
        return base_topics + worker_discoveries.topics(limit=10)

    topics = current_topics()

    for index, question in enumerate(
        topics,
        1,
    ):
        print(
            f"  {index}. {question}",
            flush=True,
        )

    print(
        "\nCommands: topics, help, exit",
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
                worker_discoveries,
                checkpoint,
                attention_controller,
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
                getattr(args, "experience_store", ""),
            )
            try:
                if checkpoint.should_sync():
                    checkpoint.sync(ram_memory, "chat", __import__("os").getpid())
            except Exception as exc:
                checkpoint.record_event("error", {"stage":"chat_checkpoint", "error":repr(exc)})
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
                "topics",
            }:
                topics = current_topics()
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
                worker_discoveries,
                checkpoint,
                attention_controller,
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
                getattr(args, "experience_store", ""),
            )
            try:
                if checkpoint.should_sync():
                    checkpoint.sync(ram_memory, "chat", __import__("os").getpid())
            except Exception as exc:
                checkpoint.record_event("error", {"stage":"chat_checkpoint", "error":repr(exc)})

    sync_stop.set()
    sync_thread.join(timeout=2.0)
    try:
        checkpoint.sync(ram_memory, "chat", __import__("os").getpid(), force=True)
    except Exception:
        pass
    checkpoint.close()

    print(
        "\n=== V679 CHAT WORKER COMPLETE ===",
        flush=True,
    )
    print(
        f"TRACE  : {trace_path}",
        flush=True,
    )
    distilled_stats = distilled.counts()
    realized_stats = graph.realized_answer_stats()
    print(
        f"DISTILLED DB: decisions={distilled_stats.get('decisions', 0)} "
        f"observations={distilled_stats.get('observations', distilled_stats.get('durable_observations', 0))}",
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



def main():
    # Standalone compatibility mode: V679 chat worker without the 19-worker runtime.
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", required=True)
    ap.add_argument("--output", default="./results/v679_chat.json")
    ap.add_argument("--trace-output", required=True)
    ap.add_argument("--memory-output", required=True)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--mode", choices=("chat","smoke"), default="chat")
    ap.add_argument("--max-hypotheses", type=int, default=12)
    ap.add_argument("--goal-budget", type=int, default=40)
    ap.add_argument("--per-node", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--cache-entries", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=67100)
    ap.add_argument("--shared-memory", default="./results/v679_shared_memory.sqlite")
    ap.add_argument("--worker-id", type=int, default=19)
    ap.add_argument("--total-workers", type=int, default=20)
    ap.add_argument("--checkpoint-seconds", type=int, choices=(60,300), default=300)
    ns=ap.parse_args()
    return run_chat_worker(ns)


if __name__ == "__main__":
    main()
