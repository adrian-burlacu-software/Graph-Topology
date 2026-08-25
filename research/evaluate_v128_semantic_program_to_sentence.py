from __future__ import annotations

"""
V128 — SEMANTIC PROGRAM -> SENTENCE

V127 showed a useful boundary:
    tiny fact sets -> clean sentence realization
    large fact sets -> prompt leakage / poor planning

V128 therefore moves planning OUT of the LLM.

Pipeline:

    ConceptNet graph
        ↓
    choose 1–2 compatible typed facts
        ↓
    deterministic semantic program / grammar frame
        ↓
    frozen SmolLM2 realizes that tiny program as one sentence

The LLM never decides:
    * which facts matter
    * which relation types are compatible
    * how many facts to combine

It only lexicalizes a tiny structured semantic program.

Tested frames:
    IsA
    IsA + CapableOf
    IsA + HasProperty
    IsA + UsedFor
    HasA
    IsA + HasA
    CapableOf
    UsedFor

Outputs:
    * generated sentence
    * facts used
    * frame used
    * clean sentence rate
    * target presence
    * fact overlap
    * generation speed
    * trace examples

No graph mutation.
No semantic learning.
"""

import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "conceptnet_compact.db"
DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
MODEL_PATH = ROOT / "llm" / "SmolLM2-360M-Instruct"
OUTPUT_PATH = ROOT / "results" / "v128_semantic_program_to_sentence.json"

EVAL_WORDS = 256
BATCH_SIZE = 64
MAX_INPUT_TOKENS = 192
MAX_NEW_TOKENS = 32
TRACE_WORDS = {
    "hello",
    "greeting",
    "dog",
    "animal",
    "ability",
    "abandon",
    "water",
    "music",
    "chair",
    "car",
}

RELATION_PRIORITY = {
    "IsA": 4.0,
    "CapableOf": 3.8,
    "HasProperty": 3.6,
    "UsedFor": 3.6,
    "HasA": 3.2,
    "PartOf": 3.2,
    "Causes": 3.0,
    "AtLocation": 2.0,
    "MadeOf": 2.0,
    "ReceivesAction": 1.5,
    "HasPrerequisite": 1.8,
    "Synonym": 1.5,
    "Antonym": 1.3,
    "RelatedTo": 0.8,
    "SimilarTo": 0.8,
    "DefinedAs": 1.5,
    "HasContext": 0.8,
}

FRAMES = (
    "IsA",
    "IsA+CapableOf",
    "IsA+HasProperty",
    "IsA+UsedFor",
    "HasA",
    "IsA+HasA",
    "CapableOf",
    "UsedFor",
)


def load_dictionary(path: Path) -> list[str]:
    words = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            word = raw.strip().lower()
            if word and word.isalpha():
                words.add(word)
    return sorted(words)[:EVAL_WORDS]


class ConceptNetDB:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def facts_for_word(self, word: str) -> list[tuple[str, str, float]]:
        rows = self.conn.execute(
            """
            SELECT relation, end, weight
            FROM edge
            WHERE start = ?
            AND relation IN (
                'IsA',
                'CapableOf',
                'HasProperty',
                'UsedFor',
                'HasA',
                'PartOf',
                'Causes',
                'AtLocation',
                'MadeOf',
                'ReceivesAction',
                'HasPrerequisite',
                'Synonym',
                'Antonym',
                'RelatedTo',
                'SimilarTo',
                'DefinedAs',
                'HasContext'
            )
            ORDER BY weight DESC, relation ASC, end ASC
            """,
            (word,),
        )
        return [
            (
                row["relation"],
                row["end"],
                float(row["weight"]),
            )
            for row in rows
        ]


def choose_best(
    facts: list[tuple[str, str, float]],
    relation: str,
) -> tuple[str, str, float] | None:
    candidates = [
        fact for fact in facts
        if fact[0] == relation
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda fact: (
            RELATION_PRIORITY.get(fact[0], 1.0) * (1.0 + fact[2])
        ),
    )


def choose_program(
    word: str,
    facts: list[tuple[str, str, float]],
) -> tuple[str, list[tuple[str, str, float]]] | None:
    """
    Deterministic semantic planner.

    Preference:
        IsA + CapableOf
        IsA + HasProperty
        IsA + UsedFor
        IsA + HasA
        IsA
        CapableOf
        UsedFor
        HasA
    """
    isa = choose_best(facts, "IsA")
    capable = choose_best(facts, "CapableOf")
    prop = choose_best(facts, "HasProperty")
    used = choose_best(facts, "UsedFor")
    hasa = choose_best(facts, "HasA")

    if isa and capable:
        return "IsA+CapableOf", [isa, capable]
    if isa and prop:
        return "IsA+HasProperty", [isa, prop]
    if isa and used:
        return "IsA+UsedFor", [isa, used]
    if isa and hasa:
        return "IsA+HasA", [isa, hasa]
    if isa:
        return "IsA", [isa]
    if capable:
        return "CapableOf", [capable]
    if used:
        return "UsedFor", [used]
    if hasa:
        return "HasA", [hasa]

    return None


def program_to_prompt_text(
    word: str,
    frame: str,
    facts: list[tuple[str, str, float]],
) -> str:
    lines = []

    for relation, end, _weight in facts:
        if relation == "IsA":
            lines.append(f"TYPE = {end}")
        elif relation == "CapableOf":
            lines.append(f"CAN = {end}")
        elif relation == "HasProperty":
            lines.append(f"PROPERTY = {end}")
        elif relation == "UsedFor":
            lines.append(f"USED_FOR = {end}")
        elif relation == "HasA":
            lines.append(f"HAS = {end}")
        elif relation == "PartOf":
            lines.append(f"PART_OF = {end}")
        else:
            lines.append(f"{relation.upper()} = {end}")

    return "\n".join(lines)


def frame_prompt(
    tokenizer,
    word: str,
    frame: str,
    facts: list[tuple[str, str, float]],
) -> str:
    semantic_program = program_to_prompt_text(
        word,
        frame,
        facts,
    )

    return apply_chat(
        tokenizer,
        (
            "You are a language realizer. "
            "Turn a tiny semantic program into one simple English sentence. "
            "Do not add facts."
        ),
        f"""
TARGET:
{word}

FRAME:
{frame}

SEMANTIC PROGRAM:
{semantic_program}

Write ONE natural English sentence expressing this program.

Rules:
- Keep TARGET as the subject when natural.
- Do not add information not present in the program.
- Do not mention the frame or semantic program.
- One sentence only.
""".strip(),
    )


def apply_chat(
    tokenizer,
    system: str,
    user: str,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return system + "\n\n" + user + "\n\nAssistant:"


def load_model():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no PAD/EOS.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to(device)

    print("device:", device, flush=True)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    return tokenizer, model, device


@torch.inference_mode()
def generate_batch(
    tokenizer,
    model,
    device,
    prompts: list[str],
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []

    for row in range(output.shape[0]):
        prompt_len = int(attention_mask[row].sum().item())
        results.append(
            tokenizer.decode(
                output[row, prompt_len:],
                skip_special_tokens=True,
            ).strip()
        )

    return results


def clean_sentence(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^(answer|response|sentence)\s*:\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    match = re.search(
        r"(.+?[.!?])(?:\s|$)",
        text,
    )
    if match:
        text = match.group(1).strip()

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def quality(
    word: str,
    sentence: str,
    facts: list[tuple[str, str, float]],
) -> dict:
    lower = sentence.lower()

    leaks = any(
        marker in lower
        for marker in (
            "semantic program",
            "graph",
            "conceptnet",
            "frame:",
            "assistant",
            "do not",
        )
    )

    has_target = word.lower() in lower
    sentence_count = len(
        re.findall(r"[.!?]+", sentence)
    )

    fact_hits = 0

    for _relation, end, _weight in facts:
        tokens = [
            token
            for token in re.findall(
                r"[a-z0-9]+",
                end.lower(),
            )
            if len(token) >= 3
        ]
        if tokens and any(token in lower for token in tokens):
            fact_hits += 1

    return {
        "has_target": has_target,
        "instruction_leak": leaks,
        "sentence_count": sentence_count,
        "fact_hits": fact_hits,
        "clean": (
            bool(sentence)
            and has_target
            and not leaks
            and sentence_count <= 2
            and len(sentence) <= 220
        ),
    }


def main() -> None:
    started = time.perf_counter()

    print("=== V128 SEMANTIC PROGRAM -> SENTENCE ===")

    words = load_dictionary(DICTIONARY_PATH)
    db = ConceptNetDB(DB_PATH)

    try:
        print("dictionary_words:", len(words), flush=True)

        programs = []

        program_counts = {}

        for word in words:
            facts = db.facts_for_word(word)
            program = choose_program(word, facts)

            if program is None:
                continue

            frame, selected_facts = program

            programs.append(
                (
                    word,
                    frame,
                    selected_facts,
                )
            )

            program_counts[frame] = (
                program_counts.get(frame, 0) + 1
            )

        print(
            "programs:",
            len(programs),
            flush=True,
        )
        print(
            "program_counts:",
            program_counts,
            flush=True,
        )

        tokenizer, model, device = load_model()

        results = {}
        trace = []

        for frame in FRAMES:
            frame_jobs = [
                item
                for item in programs
                if item[1] == frame
            ]

            if not frame_jobs:
                continue

            print(
                f"\n=== FRAME {frame} ({len(frame_jobs)}) ===",
                flush=True,
            )

            outputs = {}
            qualities = {}

            generation_started = time.perf_counter()

            for start in range(
                0,
                len(frame_jobs),
                BATCH_SIZE,
            ):
                batch = frame_jobs[
                    start:start + BATCH_SIZE
                ]

                prompts = [
                    frame_prompt(
                        tokenizer,
                        word,
                        batch_frame,
                        selected_facts,
                    )
                    for word, batch_frame, selected_facts in batch
                ]

                raw_outputs = generate_batch(
                    tokenizer,
                    model,
                    device,
                    prompts,
                )

                for (
                    (word, batch_frame, selected_facts),
                    raw,
                ) in zip(batch, raw_outputs):
                    sentence = clean_sentence(raw)

                    outputs[word] = sentence
                    qualities[word] = quality(
                        word,
                        sentence,
                        selected_facts,
                    )

                    if word in TRACE_WORDS:
                        trace.append(
                            {
                                "word": word,
                                "frame": batch_frame,
                                "facts": selected_facts,
                                "semantic_program": program_to_prompt_text(
                                    word,
                                    batch_frame,
                                    selected_facts,
                                ),
                                "raw": raw,
                                "sentence": sentence,
                                "quality": qualities[word],
                            }
                        )

                processed = min(
                    start + BATCH_SIZE,
                    len(frame_jobs),
                )

                if (
                    processed <= BATCH_SIZE
                    or processed % BATCH_SIZE == 0
                    or processed == len(frame_jobs)
                ):
                    elapsed = time.perf_counter() - generation_started
                    print(
                        f"REALIZE {processed:4d}/{len(frame_jobs):4d} "
                        f"sent/s={processed / max(0.001, elapsed):.2f}",
                        flush=True,
                    )

            clean_count = sum(
                bool(item["clean"])
                for item in qualities.values()
            )
            target_count = sum(
                bool(item["has_target"])
                for item in qualities.values()
            )
            leaks = sum(
                bool(item["instruction_leak"])
                for item in qualities.values()
            )
            fact_hits = [
                item["fact_hits"]
                for item in qualities.values()
            ]

            generation_seconds = (
                time.perf_counter()
                - generation_started
            )

            results[frame] = {
                "jobs": len(frame_jobs),
                "generation_seconds": generation_seconds,
                "sentences_per_second": (
                    len(frame_jobs)
                    / max(0.001, generation_seconds)
                ),
                "clean_sentence_rate": (
                    clean_count
                    / max(1, len(frame_jobs))
                ),
                "target_presence_rate": (
                    target_count
                    / max(1, len(frame_jobs))
                ),
                "instruction_leak_rate": (
                    leaks
                    / max(1, len(frame_jobs))
                ),
                "mean_fact_hits": (
                    sum(fact_hits)
                    / max(1, len(fact_hits))
                ),
                "sentences": outputs,
                "quality": qualities,
            }

        print("\n=== V128 SUMMARY ===")
        print(
            "frame | jobs | sent/s | clean | target | leak | fact_hits"
        )

        for frame, result in results.items():
            print(
                f"{frame:18s} | "
                f"{result['jobs']:4d} | "
                f"{result['sentences_per_second']:.2f} | "
                f"{result['clean_sentence_rate']:.4f} | "
                f"{result['target_presence_rate']:.4f} | "
                f"{result['instruction_leak_rate']:.4f} | "
                f"{result['mean_fact_hits']:.2f}"
            )

        print("\n=== TRACE ===")
        for item in trace:
            print(json.dumps(item, ensure_ascii=False))

        report = {
            "experiment": "V128 semantic program -> sentence",
            "dictionary_words": len(words),
            "programs": len(programs),
            "program_counts": program_counts,
            "frames": FRAMES,
            "results": results,
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "\nsaved:",
            OUTPUT_PATH,
        )
        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )
        print("=== V128 COMPLETE ===")

    finally:
        db.close()


if __name__ == "__main__":
    main()
