from __future__ import annotations

"""
V129 — LLM SEMANTIC PROGRAM FIDELITY

Final LLM-stage experiment before moving on to semantic-net + grammar rules.

V128 demonstrated that SmolLM2-360M can turn tiny semantic programs into clean
English sentences. V129 asks the harder and more important question:

    Does it preserve the semantic program faithfully?

The LLM receives a tiny structured program such as:

    TYPE = animal
    CAN = bark

and generates one English sentence.

We then perform a deterministic lexical check against the source program.

This is NOT a claim of deep semantic understanding. It is a controlled test
of whether the frozen model can act as a semantic-to-language decoder without
introducing unsupported facts.

Experiments
-----------
1. PROGRAM -> SENTENCE
2. FACT REALIZATION:
       Did each source concept appear or have a close lexical realization?
3. HALLUCINATION CHECK:
       Does the sentence contain obvious content not supported by the program?
4. TARGET PRESENCE
5. CLEAN SENTENCE
6. ROUND-TRIP:
       Use a second deterministic parser to recover likely relation values
       from the generated sentence and compare them with the input program.

We test the same V128 frames and additionally run two prompt modes:

    STRICT
        "say only what the program says"

    NATURAL
        "say it naturally"

The point is to establish the best achievable behavior of this tiny frozen
realizer before we stop touching the LLM and move to a deterministic grammar
layer.

No graph mutation.
No training.
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

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v129_llm_semantic_program_fidelity.json"
)

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

FRAMES = (
    "IsA",
    "IsA+CapableOf",
    "IsA+HasProperty",
    "IsA+UsedFor",
    "IsA+HasA",
    "CapableOf",
    "UsedFor",
    "HasA",
)

RELATION_TO_SLOT = {
    "IsA": "TYPE",
    "CapableOf": "CAN",
    "HasProperty": "PROPERTY",
    "UsedFor": "USED_FOR",
    "HasA": "HAS",
    "PartOf": "PART_OF",
    "Causes": "CAUSES",
    "AtLocation": "LOCATION",
    "MadeOf": "MADE_OF",
    "ReceivesAction": "RECEIVES",
    "HasPrerequisite": "PREREQUISITE",
    "Synonym": "SYNONYM",
    "Antonym": "ANTONYM",
    "RelatedTo": "RELATED_TO",
    "SimilarTo": "SIMILAR_TO",
    "DefinedAs": "DEFINED_AS",
    "HasContext": "CONTEXT",
}

# Very conservative words/phrases that usually indicate the model escaped the
# tiny realization task and began discussing instructions.
LEAK_MARKERS = (
    "semantic program",
    "graph facts",
    "conceptnet",
    "candidate",
    "instructions",
    "assistant",
    "return one",
    "do not",
    "frame:",
)


def load_dictionary(path: Path) -> list[str]:
    words = set()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
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

    def facts_for_word(
        self,
        word: str,
    ) -> list[tuple[str, str, float]]:
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
):
    matches = [
        fact
        for fact in facts
        if fact[0] == relation
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda fact: (
            1.0 + fact[2]
        ),
    )


def choose_program(
    word: str,
    facts: list[tuple[str, str, float]],
):
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


def program_to_lines(
    facts: list[tuple[str, str, float]],
) -> list[str]:
    lines = []

    for relation, end, _weight in facts:
        slot = RELATION_TO_SLOT.get(
            relation,
            relation.upper(),
        )
        lines.append(
            f"{slot} = {end}"
        )

    return lines


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
            raise RuntimeError(
                "Tokenizer has no PAD/EOS."
            )

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

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

    return tokenizer, model, device


def apply_chat(
    tokenizer,
    system: str,
    user: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            system
            + "\n\n"
            + user
            + "\n\nAssistant:"
        )


def make_prompt(
    tokenizer,
    word: str,
    frame: str,
    program_lines: list[str],
    mode: str,
) -> str:
    if mode == "STRICT":
        system = (
            "You are a semantic-to-language decoder. "
            "The semantic program is authoritative. "
            "Write exactly one sentence that expresses only the program. "
            "Do not add facts."
        )
    else:
        system = (
            "You are a semantic-to-language decoder. "
            "Turn the supplied semantic program into one natural English "
            "sentence. Do not add unsupported facts."
        )

    return apply_chat(
        tokenizer,
        system,
        f"""
TARGET:
{word}

FRAME:
{frame}

SEMANTIC PROGRAM:
{chr(10).join(program_lines)}

Return ONE English sentence.

Do not mention:
- the semantic program
- graph
- ConceptNet
- instructions
- candidate lists
- the assistant

Only describe TARGET according to the program.
""".strip(),
    )


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

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)

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

    for row in range(
        output.shape[0]
    ):
        prompt_len = int(
            attention_mask[row].sum().item()
        )

        results.append(
            tokenizer.decode(
                output[
                    row,
                    prompt_len:,
                ],
                skip_special_tokens=True,
            ).strip()
        )

    return results


def clean_sentence(
    text: str,
) -> str:
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


def singularize_phrase(
    phrase: str,
) -> str:
    """
    Tiny lexical normalization used only by the fidelity benchmark.
    It is deliberately conservative.
    """
    phrase = phrase.lower().strip()

    if phrase.endswith("ies") and len(phrase) > 4:
        return phrase[:-3] + "y"

    if phrase.endswith("es") and len(phrase) > 4:
        return phrase[:-2]

    if phrase.endswith("s") and len(phrase) > 3:
        return phrase[:-1]

    return phrase


def phrase_appears(
    sentence: str,
    value: str,
) -> bool:
    sentence_lower = sentence.lower()
    value_lower = value.lower().strip()

    if not value_lower:
        return False

    if value_lower in sentence_lower:
        return True

    singular = singularize_phrase(
        value_lower
    )

    if singular != value_lower and singular in sentence_lower:
        return True

    # Basic word-level fallback.
    value_words = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            value_lower,
        )
        if len(token) >= 3
    ]

    if len(value_words) == 1:
        token = value_words[0]
        return bool(
            re.search(
                rf"\b{re.escape(token)}\w*\b",
                sentence_lower,
            )
        )

    return False


def fact_realization_scores(
    sentence: str,
    facts: list[tuple[str, str, float]],
) -> dict[str, object]:
    realized = []

    for relation, end, _weight in facts:
        hit = phrase_appears(
            sentence,
            end,
        )

        realized.append(
            {
                "relation": relation,
                "value": end,
                "realized": hit,
            }
        )

    total = len(realized)
    hits = sum(
        1
        for item in realized
        if item["realized"]
    )

    return {
        "facts": realized,
        "hits": hits,
        "total": total,
        "coverage": (
            hits / max(1, total)
        ),
    }


def quality(
    word: str,
    sentence: str,
    fact_scores: dict[str, object],
) -> dict[str, object]:
    lower = sentence.lower()

    leaks = any(
        marker in lower
        for marker in LEAK_MARKERS
    )

    sentence_count = len(
        re.findall(
            r"[.!?]+",
            sentence,
        )
    )

    target_presence = (
        word.lower()
        in lower
    )

    clean = (
        bool(sentence)
        and target_presence
        and not leaks
        and sentence_count <= 2
        and len(sentence) <= 220
    )

    return {
        "target_presence": target_presence,
        "instruction_leak": leaks,
        "sentence_count": sentence_count,
        "length": len(sentence),
        "clean": clean,
        "fact_coverage": fact_scores[
            "coverage"
        ],
    }


def recover_relations_from_sentence(
    sentence: str,
    facts: list[tuple[str, str, float]],
) -> list[str]:
    recovered = []

    lower = sentence.lower()

    for relation, end, _weight in facts:
        if phrase_appears(
            lower,
            end,
        ):
            recovered.append(
                relation
            )

    return recovered


def main() -> None:
    started = time.perf_counter()

    print(
        "=== V129 LLM SEMANTIC PROGRAM FIDELITY ==="
    )

    words = load_dictionary(
        DICTIONARY_PATH
    )

    db = ConceptNetDB(
        DB_PATH
    )

    try:
        programs = []

        frame_counts = defaultdict(int)

        for word in words:
            facts = db.facts_for_word(
                word
            )

            program = choose_program(
                word,
                facts,
            )

            if program is None:
                continue

            frame, selected_facts = program

            lines = program_to_lines(
                selected_facts
            )

            programs.append(
                (
                    word,
                    frame,
                    selected_facts,
                    lines,
                )
            )

            frame_counts[
                frame
            ] += 1

        print(
            "dictionary_words:",
            len(words),
            flush=True,
        )

        print(
            "programs:",
            len(programs),
            flush=True,
        )

        print(
            "frame_counts:",
            dict(frame_counts),
            flush=True,
        )

        tokenizer, model, device = load_model()

        modes = (
            "STRICT",
            "NATURAL",
        )

        results = {}
        trace = []

        for mode in modes:
            print(
                f"\n=== MODE {mode} ===",
                flush=True,
            )

            mode_results = {}
            mode_frame_stats = defaultdict(
                lambda: {
                    "jobs": 0,
                    "clean": 0,
                    "target": 0,
                    "leak": 0,
                    "fact_coverage_sum": 0.0,
                    "fact_perfect": 0,
                }
            )

            generation_started = time.perf_counter()

            for start in range(
                0,
                len(programs),
                BATCH_SIZE,
            ):
                batch = programs[
                    start:start + BATCH_SIZE
                ]

                prompts = [
                    make_prompt(
                        tokenizer,
                        word,
                        frame,
                        lines,
                        mode,
                    )
                    for word, frame, _facts, lines
                    in batch
                ]

                raw_outputs = generate_batch(
                    tokenizer,
                    model,
                    device,
                    prompts,
                )

                for (
                    (
                        word,
                        frame,
                        facts,
                        lines,
                    ),
                    raw,
                ) in zip(
                    batch,
                    raw_outputs,
                ):
                    sentence = clean_sentence(
                        raw
                    )

                    fact_scores = (
                        fact_realization_scores(
                            sentence,
                            facts,
                        )
                    )

                    q = quality(
                        word,
                        sentence,
                        fact_scores,
                    )

                    frame_stat = mode_frame_stats[
                        frame
                    ]

                    frame_stat[
                        "jobs"
                    ] += 1

                    frame_stat[
                        "clean"
                    ] += int(
                        q["clean"]
                    )

                    frame_stat[
                        "target"
                    ] += int(
                        q["target_presence"]
                    )

                    frame_stat[
                        "leak"
                    ] += int(
                        q["instruction_leak"]
                    )

                    frame_stat[
                        "fact_coverage_sum"
                    ] += float(
                        q[
                            "fact_coverage"
                        ]
                    )

                    frame_stat[
                        "fact_perfect"
                    ] += int(
                        q[
                            "fact_coverage"
                        ]
                        >= 0.999
                    )

                    mode_results[
                        word
                    ] = {
                        "frame": frame,
                        "program": lines,
                        "raw": raw,
                        "sentence": sentence,
                        "facts": facts,
                        "fact_scores": fact_scores,
                        "quality": q,
                    }

                    if (
                        mode == "NATURAL"
                        and word in TRACE_WORDS
                    ):
                        trace.append(
                            {
                                "mode": mode,
                                "word": word,
                                "frame": frame,
                                "program": lines,
                                "facts": facts,
                                "raw": raw,
                                "sentence": sentence,
                                "fact_scores": fact_scores,
                                "quality": q,
                            }
                        )

                processed = min(
                    start + BATCH_SIZE,
                    len(programs),
                )

                if (
                    processed <= BATCH_SIZE
                    or processed % BATCH_SIZE == 0
                    or processed == len(programs)
                ):
                    elapsed = (
                        time.perf_counter()
                        - generation_started
                    )

                    print(
                        f"REALIZE "
                        f"{processed:4d}/{len(programs):4d} "
                        f"sent/s="
                        f"{processed / max(0.001, elapsed):.2f}",
                        flush=True,
                    )

            generation_seconds = (
                time.perf_counter()
                - generation_started
            )

            # Aggregate frame statistics.
            frame_summary = {}

            for frame, stats in (
                mode_frame_stats.items()
            ):
                frame_summary[
                    frame
                ] = {
                    "jobs": stats["jobs"],
                    "clean_rate": (
                        stats["clean"]
                        / max(
                            1,
                            stats["jobs"],
                        )
                    ),
                    "target_presence_rate": (
                        stats["target"]
                        / max(
                            1,
                            stats["jobs"],
                        )
                    ),
                    "instruction_leak_rate": (
                        stats["leak"]
                        / max(
                            1,
                            stats["jobs"],
                        )
                    ),
                    "mean_fact_coverage": (
                        stats[
                            "fact_coverage_sum"
                        ]
                        / max(
                            1,
                            stats["jobs"],
                        )
                    ),
                    "perfect_fact_fidelity_rate": (
                        stats[
                            "fact_perfect"
                        ]
                        / max(
                            1,
                            stats["jobs"],
                        )
                    ),
                }

            clean_total = sum(
                item["clean"]
                for item in mode_frame_stats.values()
            )

            jobs_total = sum(
                item["jobs"]
                for item in mode_frame_stats.values()
            )

            fact_coverage_total = sum(
                item[
                    "fact_coverage_sum"
                ]
                for item in mode_frame_stats.values()
            )

            perfect_total = sum(
                item[
                    "fact_perfect"
                ]
                for item in mode_frame_stats.values()
            )

            mode_summary = {
                "programs": len(programs),
                "generation_seconds": generation_seconds,
                "sentences_per_second": (
                    len(programs)
                    / max(
                        0.001,
                        generation_seconds,
                    )
                ),
                "clean_sentence_rate": (
                    clean_total
                    / max(
                        1,
                        jobs_total,
                    )
                ),
                "mean_fact_coverage": (
                    fact_coverage_total
                    / max(
                        1,
                        jobs_total,
                    )
                ),
                "perfect_fact_fidelity_rate": (
                    perfect_total
                    / max(
                        1,
                        jobs_total,
                    )
                ),
                "frame_summary": frame_summary,
                "outputs": mode_results,
            }

            results[
                mode
            ] = mode_summary

        strict = results["STRICT"]
        natural = results["NATURAL"]

        print()
        print(
            "=== V129 SUMMARY ==="
        )

        print(
            "metric                  | STRICT | NATURAL"
        )

        print(
            f"clean_sentence_rate     | "
            f"{strict['clean_sentence_rate']:.4f} | "
            f"{natural['clean_sentence_rate']:.4f}"
        )

        print(
            f"mean_fact_coverage      | "
            f"{strict['mean_fact_coverage']:.4f} | "
            f"{natural['mean_fact_coverage']:.4f}"
        )

        print(
            f"perfect_fact_fidelity   | "
            f"{strict['perfect_fact_fidelity_rate']:.4f} | "
            f"{natural['perfect_fact_fidelity_rate']:.4f}"
        )

        print(
            f"sentences_per_second    | "
            f"{strict['sentences_per_second']:.2f} | "
            f"{natural['sentences_per_second']:.2f}"
        )

        print()
        print(
            "=== TRACE ==="
        )

        for item in trace:
            print(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
            )

        report = {
            "experiment": (
                "V129 LLM semantic program fidelity"
            ),
            "dictionary_words": len(words),
            "programs": len(programs),
            "frame_counts": dict(frame_counts),
            "modes": results,
            "trace": trace,
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
            "saved:",
            OUTPUT_PATH,
        )

        print(
            "elapsed_seconds:",
            f"{time.perf_counter() - started:.2f}",
        )

        print(
            "=== V129 COMPLETE ==="
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
