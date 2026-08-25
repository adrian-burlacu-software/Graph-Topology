from __future__ import annotations

"""
V106 — LIVE LLM ACTIVATION TRAJECTORY MOTIFS

LAST LLM EXPERIMENT

V105 showed that whole-word activation vectors do not naturally collapse into
many near-duplicate prototypes.

So this final attempt changes the unit of analysis:

    whole-word vector            X
    token activation trajectory  YES

For each dictionary word:

    tokens
      ↓
    layer-3 token hidden states
      ↓
    consecutive activation deltas
      ↓
    normalized delta directions
      ↓
    discrete direction signatures
      ↓
    reusable trajectory motifs

The key question:

    Do recurring activation-transition motifs form a reusable graph that
    preserves or predicts independently observed semantic structure?

Important:
    * no saved PT activation file is used
    * the local SmolLM2-360M is loaded live
    * the entire dictionary is processed
    * semantics-large.csv is evaluation-only
    * no target number of motifs is chosen
    * no nearest-neighbor ordering is invented
    * no prototype hierarchy is imposed

Motif signature:
    A deterministic sign hash of normalized activation-delta directions.

A motif is "reused" simply when the same signature occurs in multiple words.

Per-word representation:
    set of motif IDs + counts

Semantic validation:
    correlation between motif-set similarity and human feature-set similarity

Null:
    shuffled semantic assignments are not necessary for the core topology test;
    the decisive question here is whether semantic similarity tracks motif
    similarity above a permutation baseline.

The permutation baseline shuffles semantic feature sets across words while
leaving the activation motif topology untouched.

If motif similarity has meaningful semantic correlation and motif reuse is
substantial, the LLM is providing a discrete reusable trajectory substrate.
"""

import csv
import hashlib
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "llm"
    / "SmolLM2-360M"
)

DICTIONARY_PATH = (
    ROOT
    / "data"
    / "dictionary.csv"
)

SEMANTICS_PATH = (
    ROOT
    / "data"
    / "semantics-large.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "v106_llm_activation_trajectory_motifs.pt"
)

TARGET_LAYER = 3
BATCH_SIZE = 16

# Number of random hyperplanes used for the direction signature.
# This is a representation width, not a number of motifs.
SIGNATURE_BITS = 32
SIGNATURE_SEED = 9173

# Ignore extremely short token sequences.
MIN_TOKENS_FOR_TRAJECTORY = 2

# Semantic nearest-neighbor sample.
SEMANTIC_NEIGHBORS = 20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(
        torch.backends,
        "mps",
        None,
    ) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def load_words(
    path: Path,
) -> list[str]:
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

    result = sorted(words)

    if not result:
        raise RuntimeError(
            "No dictionary words found."
        )

    return result


def load_semantics(
    path: Path,
) -> dict[str, dict[str, float]]:
    records: dict[
        str,
        dict[str, float],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "cue",
            "translated",
            "frequency_translated",
            "normalized_translated",
            "n",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "semantics-large.csv missing: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            cue = row["cue"].strip().lower()
            feature = row["translated"].strip().lower()

            if not cue or not feature:
                continue

            try:
                weight = float(
                    row["normalized_translated"]
                )
            except (
                TypeError,
                ValueError,
            ):
                weight = 0.0

            if weight <= 0.0:
                try:
                    frequency = float(
                        row["frequency_translated"]
                    )
                    n = int(row["n"])
                except (
                    TypeError,
                    ValueError,
                ):
                    frequency = 0.0
                    n = 0

                if n > 0:
                    weight = frequency / n

            if weight <= 0.0:
                continue

            records.setdefault(
                cue,
                {},
            )

            records[
                cue
            ][feature] = (
                records[cue].get(
                    feature,
                    0.0,
                )
                + weight
            )

    return records


def load_model():
    device = choose_device()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has no pad/eos token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Representation cleanup
# ---------------------------------------------------------------------------

def clean_word_level_common_component(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the V102 cleanup to each token activation matrix in-place at the
    batch level:

        center across all observed tokens
        remove top common component

    Here the common component is estimated from the full token sample for the
    whole dictionary.
    """
    mean = x.mean(
        dim=0
    )

    centered = x - mean

    q = min(
        4,
        centered.shape[1] - 1,
        centered.shape[0] - 1,
    )

    if q <= 0:
        return centered

    _, _, v = torch.pca_lowrank(
        centered.cpu(),
        q=q,
        center=False,
    )

    component = v[:, 0].unsqueeze(
        0
    )

    corrected = (
        centered.cpu()
        - (
            centered.cpu()
            @ component.T
        )
        @ component
    )

    return corrected.float()


# ---------------------------------------------------------------------------
# Signature space
# ---------------------------------------------------------------------------

def build_projection(
    dimension: int,
) -> torch.Tensor:
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SIGNATURE_SEED
    )

    projection = torch.randn(
        (
            SIGNATURE_BITS,
            dimension,
        ),
        generator=generator,
        dtype=torch.float32,
    )

    projection = projection / torch.linalg.vector_norm(
        projection,
        dim=1,
        keepdim=True,
    ).clamp_min(
        1e-12
    )

    return projection


def direction_signature(
    delta: torch.Tensor,
    projection: torch.Tensor,
) -> int:
    normalized = delta / torch.linalg.vector_norm(
        delta
    ).clamp_min(
        1e-12
    )

    scores = projection @ normalized

    bits = (
        scores >= 0
    )

    signature = 0

    for index, flag in enumerate(
        bits.tolist()
    ):
        if flag:
            signature |= (
                1 << index
            )

    return signature


# ---------------------------------------------------------------------------
# Motif extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_token_trajectory_motifs(
    tokenizer,
    model,
    device,
    words: list[str],
) -> tuple[
    dict[str, Counter[int]],
    Counter[int],
    int,
]:
    """
    Returns:
        word -> motif Counter
        global motif occurrence count
        hidden dimension
    """
    # Probe dimension.
    probe = tokenizer(
        words[:1],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    probe_out = model(
        input_ids=probe["input_ids"].to(device),
        attention_mask=probe["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_size = int(
        probe_out.hidden_states[
            TARGET_LAYER
        ].shape[-1]
    )

    projection = build_projection(
        hidden_size
    )

    # First pass: gather a representative token-state sample to estimate
    # the common direction. We use the first 1024 words to keep the estimate
    # cheap, then apply it to every trajectory.
    sample_states = []

    sample_words = words[
        :min(
            1024,
            len(words),
        )
    ]

    for start in range(
        0,
        len(sample_words),
        BATCH_SIZE,
    ):
        batch_words = sample_words[
            start:start + BATCH_SIZE
        ]

        batch = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            use_cache=False,
        )

        hidden = output.hidden_states[
            TARGET_LAYER
        ].float().cpu()

        mask = batch[
            "attention_mask"
        ].bool()

        sample_states.append(
            hidden[
                mask
            ]
        )

    sample = torch.cat(
        sample_states,
        dim=0,
    )

    sample_clean = clean_word_level_common_component(
        sample
    )

    # One global centering vector for all trajectories.
    global_mean = sample_clean.mean(
        dim=0
    )

    word_motifs: dict[
        str,
        Counter[int],
    ] = {}

    global_occurrences = Counter()

    total_trajectories = 0

    # Second pass: extract each word's token trajectory.
    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        batch = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            output_hidden_states=True,
            use_cache=False,
        )

        hidden = output.hidden_states[
            TARGET_LAYER
        ].float().cpu()

        masks = batch[
            "attention_mask"
        ].bool()

        for row, word in enumerate(
            batch_words
        ):
            states = hidden[
                row
            ][
                masks[row]
            ]

            if states.shape[0] < (
                MIN_TOKENS_FOR_TRAJECTORY
            ):
                continue

            # Apply the global centering direction.
            states = states - global_mean

            deltas = (
                states[1:]
                - states[:-1]
            )

            motifs = Counter()

            for delta in deltas:
                signature = direction_signature(
                    delta,
                    projection,
                )

                motifs[signature] += 1
                global_occurrences[
                    signature
                ] += 1

            word_motifs[word] = motifs
            total_trajectories += 1

        if (
            start == 0
            or start + BATCH_SIZE
            >= len(words)
        ):
            print(
                "TRAJECTORIES:",
                min(
                    start + BATCH_SIZE,
                    len(words),
                ),
                "/",
                len(words),
                flush=True,
            )

    return (
        word_motifs,
        global_occurrences,
        hidden_size,
    )


# ---------------------------------------------------------------------------
# Semantic similarity from motif topology
# ---------------------------------------------------------------------------

def motif_jaccard(
    a: Counter[int],
    b: Counter[int],
) -> float:
    keys = set(a) | set(b)

    if not keys:
        return 0.0

    num = 0.0
    den = 0.0

    for key in keys:
        av = a.get(key, 0)
        bv = b.get(key, 0)

        num += min(
            av,
            bv,
        )

        den += max(
            av,
            bv,
        )

    return num / max(
        1e-12,
        den,
    )


def weighted_semantic_jaccard(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    keys = set(a) | set(b)

    if not keys:
        return 0.0

    num = 0.0
    den = 0.0

    for key in keys:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)

        num += min(
            av,
            bv,
        )

        den += max(
            av,
            bv,
        )

    return num / max(
        1e-12,
        den,
    )


def rank_corr(
    predicted: list[float],
    target: list[float],
) -> float:
    if len(predicted) < 20:
        return 0.0

    p = torch.tensor(
        predicted,
        dtype=torch.float32,
    )

    t = torch.tensor(
        target,
        dtype=torch.float32,
    )

    p_order = torch.argsort(
        p
    )

    t_order = torch.argsort(
        t
    )

    p_rank = torch.empty_like(
        p
    )

    t_rank = torch.empty_like(
        t
    )

    p_rank[
        p_order
    ] = torch.arange(
        p.numel(),
        dtype=p.dtype,
    )

    t_rank[
        t_order
    ] = torch.arange(
        t.numel(),
        dtype=t.dtype,
    )

    p_rank -= p_rank.mean()
    t_rank -= t_rank.mean()

    denominator = (
        torch.linalg.vector_norm(
            p_rank
        )
        * torch.linalg.vector_norm(
            t_rank
        )
    ).clamp_min(
        1e-12
    )

    return float(
        torch.dot(
            p_rank,
            t_rank,
        ).item()
        / denominator.item()
    )


def motif_semantic_alignment(
    words: list[str],
    word_motifs: dict[str, Counter[int]],
    semantics: dict[str, dict[str, float]],
) -> dict[str, float]:
    anchors = [
        word
        for word in semantics
        if word in word_motifs
    ]

    predicted = []
    target = []

    pair_count = 0

    for index, word in enumerate(
        anchors
    ):
        # Compare to semantic neighbors only.
        candidate_scores = []

        for other in anchors:
            if other == word:
                continue

            similarity = motif_jaccard(
                word_motifs[word],
                word_motifs[other],
            )

            candidate_scores.append(
                (
                    similarity,
                    other,
                )
            )

        candidate_scores.sort(
            reverse=True
        )

        for similarity, other in candidate_scores[
            :SEMANTIC_NEIGHBORS
        ]:
            predicted.append(
                similarity
            )

            target.append(
                weighted_semantic_jaccard(
                    semantics[word],
                    semantics[other],
                )
            )

            pair_count += 1

    return {
        "anchors": float(
            len(anchors)
        ),
        "pairs": float(
            pair_count
        ),
        "spearman": rank_corr(
            predicted,
            target,
        ),
    }


def permutation_control(
    words: list[str],
    word_motifs: dict[str, Counter[int]],
    semantics: dict[str, dict[str, float]],
) -> float:
    """
    One deterministic permutation baseline.
    """
    anchors = [
        word
        for word in semantics
        if word in word_motifs
    ]

    semantic_sets = [
        semantics[word]
        for word in anchors
    ]

    rng = random.Random(
        SIGNATURE_SEED
    )

    rng.shuffle(
        semantic_sets
    )

    shuffled = {
        word: features
        for word, features
        in zip(
            anchors,
            semantic_sets,
        )
    }

    predicted = []
    target = []

    for word in anchors:
        candidates = []

        for other in anchors:
            if other == word:
                continue

            candidates.append(
                (
                    motif_jaccard(
                        word_motifs[word],
                        word_motifs[other],
                    ),
                    other,
                )
            )

        candidates.sort(
            reverse=True
        )

        for motif_similarity, other in candidates[
            :SEMANTIC_NEIGHBORS
        ]:
            predicted.append(
                motif_similarity
            )

            target.append(
                weighted_semantic_jaccard(
                    shuffled[word],
                    shuffled[other],
                )
            )

    return rank_corr(
        predicted,
        target,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V106 LIVE LLM ACTIVATION TRAJECTORY MOTIFS ==="
    )
    print(
        "Final LLM attempt: token-level activation transitions."
    )
    print()

    words = load_words(
        DICTIONARY_PATH
    )

    semantics = load_semantics(
        SEMANTICS_PATH
    )

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "semantic_cues:",
        len(semantics),
    )

    print()

    tokenizer, model, device = (
        load_model()
    )

    (
        word_motifs,
        global_occurrences,
        hidden_size,
    ) = extract_token_trajectory_motifs(
        tokenizer,
        model,
        device,
        words,
    )

    print()
    print(
        "=== MOTIF TOPOLOGY ==="
    )

    print(
        "words_with_trajectories:",
        len(word_motifs),
    )

    print(
        "hidden_size:",
        hidden_size,
    )

    print(
        "unique_direction_motifs:",
        len(global_occurrences),
    )

    reused_motifs = [
        motif
        for motif, count
        in global_occurrences.items()
        if count >= 2
    ]

    print(
        "reused_direction_motifs:",
        len(reused_motifs),
    )

    total_occurrences = sum(
        global_occurrences.values()
    )

    print(
        "total_motif_occurrences:",
        total_occurrences,
    )

    print(
        "motif_reuse_fraction:",
        len(reused_motifs)
        / max(
            1,
            len(global_occurrences),
        ),
    )

    if global_occurrences:
        print(
            "max_motif_reuse:",
            max(
                global_occurrences.values()
            ),
        )

    print()

    # Distribution of trajectory motif vocabulary per word.
    motif_counts_per_word = [
        len(motifs)
        for motifs
        in word_motifs.values()
    ]

    print(
        "mean_motifs_per_word:",
        sum(
            motif_counts_per_word
        )
        / max(
            1,
            len(
                motif_counts_per_word
            ),
        ),
    )

    print()

    # ---------------------------------------------------------------
    # Semantic alignment.
    # ---------------------------------------------------------------

    alignment = motif_semantic_alignment(
        words,
        word_motifs,
        semantics,
    )

    shuffled = permutation_control(
        words,
        word_motifs,
        semantics,
    )

    print(
        "=== V106 SEMANTIC ALIGNMENT ==="
    )

    print(
        "anchors:",
        alignment["anchors"],
    )

    print(
        "pairs:",
        alignment["pairs"],
    )

    print(
        "motif_semantic_spearman:",
        alignment["spearman"],
    )

    print(
        "shuffled_spearman:",
        shuffled,
    )

    print(
        "real_minus_shuffled:",
        alignment["spearman"]
        - shuffled,
    )

    print()

    # ---------------------------------------------------------------
    # Recurring motif examples.
    # ---------------------------------------------------------------

    most_reused = sorted(
        global_occurrences.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:20]

    print(
        "=== MOST REUSED DIRECTION MOTIFS ==="
    )

    for signature, count in most_reused:
        owners = [
            word
            for word, motifs
            in word_motifs.items()
            if signature in motifs
        ][:8]

        print(
            f"signature={signature:10d} "
            f"occurrences={count:4d} "
            f"words={owners}"
        )

    print()

    # ---------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------

    torch.save(
        {
            "words": words,
            "word_motifs": dict(
                word_motifs
            ),
            "global_motif_occurrences": dict(
                global_occurrences
            ),
            "semantic_alignment": alignment,
            "shuffled_spearman": shuffled,
            "config": {
                "target_layer": TARGET_LAYER,
                "signature_bits": SIGNATURE_BITS,
                "signature_seed": SIGNATURE_SEED,
                "min_tokens_for_trajectory": (
                    MIN_TOKENS_FOR_TRAJECTORY
                ),
            },
        },
        OUTPUT_PATH,
    )

    print(
        "saved:",
        OUTPUT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V106 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
