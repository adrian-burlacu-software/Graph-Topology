from __future__ import annotations

"""
V105 — LIVE LLM NATURAL ACTIVATION TOPOLOGY

This is the "stop forcing the hierarchy" experiment.

Input:
    ./llm/SmolLM2-360M

Data:
    ALL words in data/dictionary.csv
    semantics-large.csv for independent semantic validation

Representation:
    SmolLM2 layer 3
    mean-centered
    top common component removed

What is different from V104B:
    We do NOT impose 512 -> 128 -> 32 -> 8 -> 4.

Instead:
    1. discover activation prototypes using an adaptive radius
    2. require reuse (prototype must represent >= MIN_CLUSTER_SIZE words)
    3. recursively compress only when the next level produces genuine reuse
    4. stop automatically at a fixed point

Prototype discovery:
    greedy cosine-radius covering

For each unassigned activation vector:
    make it the seed of a prototype
    absorb all remaining vectors whose cosine similarity exceeds the radius
    keep prototypes that are genuinely reused
    otherwise the residual vector stays as a singleton prototype

This gives an observed natural prototype vocabulary rather than an externally
chosen K.

The recursive level operates on prototype centroids using the same natural
radius rule.

Primary outputs:
    natural prototype count
    singleton count
    reuse distribution
    actual compression
    fixed-point level
    semantic coherence
    activation->semantic correlation
"""

import csv
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
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
    / "v105_live_llm_natural_activation_topology.pt"
)

TARGET_LAYER = 3
REMOVE_TOP_COMPONENTS = 1

BATCH_SIZE = 16

# Natural topology controls.
# These are not counts of prototypes.
COSINE_RADIUS = 0.92
MIN_CLUSTER_SIZE = 2

MAX_LEVELS = 12

SEMANTIC_NEIGHBORS = 20


# ---------------------------------------------------------------------------
# Model / corpus
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
            "No dictionary words."
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


@torch.no_grad()
def extract_layer3(
    tokenizer,
    model,
    device,
    words: list[str],
) -> torch.Tensor:
    batches = []

    for start in range(
        0,
        len(words),
        BATCH_SIZE,
    ):
        batch_words = words[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_words,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        input_ids = encoded[
            "input_ids"
        ].to(device)

        attention_mask = encoded[
            "attention_mask"
        ].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden = output.hidden_states[
            TARGET_LAYER
        ].float()

        mask = attention_mask.unsqueeze(
            -1
        ).float()

        pooled = (
            hidden * mask
        ).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1.0)

        batches.append(
            pooled.cpu()
        )

        if (
            start == 0
            or start + BATCH_SIZE
            >= len(words)
        ):
            print(
                "EXTRACTED:",
                min(
                    start + BATCH_SIZE,
                    len(words),
                ),
                "/",
                len(words),
                flush=True,
            )

    raw = torch.cat(
        batches,
        dim=0,
    ).float()

    mean = raw.mean(
        dim=0
    )

    centered = raw - mean

    work = centered.cpu()

    q = min(
        REMOVE_TOP_COMPONENTS + 4,
        work.shape[1] - 1,
        work.shape[0] - 1,
    )

    if q <= 0:
        return centered

    _, _, v = torch.pca_lowrank(
        work,
        q=q,
        center=False,
    )

    components = v[
        :,
        :REMOVE_TOP_COMPONENTS,
    ].transpose(
        0,
        1,
    )

    corrected = (
        work
        - (
            work @ components.transpose(
                0,
                1,
            )
        )
        @ components
    )

    return corrected.float()


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------

def weighted_jaccard(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    keys = set(a) | set(b)

    if not keys:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for key in keys:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)

        numerator += min(av, bv)
        denominator += max(av, bv)

    return numerator / max(
        1e-12,
        denominator,
    )


# ---------------------------------------------------------------------------
# Natural prototype discovery
# ---------------------------------------------------------------------------

def natural_cover(
    vectors: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Greedy cosine-radius covering.

    Returns:
        assignments [N]
        normalized centroids [K,D]
        cluster counts [K]

    No target K is given.
    """
    x = F.normalize(
        vectors.float(),
        dim=1,
    )

    n = x.shape[0]

    unassigned = torch.ones(
        n,
        dtype=torch.bool,
    )

    assignments = torch.full(
        (n,),
        -1,
        dtype=torch.long,
    )

    centroids = []
    counts = []

    cluster_id = 0

    while bool(
        unassigned.any()
    ):
        remaining = torch.where(
            unassigned
        )[0]

        seed = int(
            remaining[0].item()
        )

        seed_vector = x[seed]

        similarities = (
            x[remaining]
            @ seed_vector
        )

        selected = remaining[
            similarities
            >= COSINE_RADIUS
        ]

        # Guarantee at least the seed.
        if len(selected) == 0:
            selected = remaining[:1]

        centroid = x[
            selected
        ].mean(
            dim=0
        )

        centroid = F.normalize(
            centroid.unsqueeze(0),
            dim=1,
        )[0]

        size = len(selected)

        for index in selected:
            assignments[
                index
            ] = cluster_id

        unassigned[
            selected
        ] = False

        centroids.append(
            centroid
        )

        counts.append(
            size
        )

        cluster_id += 1

        if (
            cluster_id % 250 == 0
            or not bool(
                unassigned.any()
            )
        ):
            print(
                f"      prototypes={cluster_id:5d} "
                f"remaining={int(unassigned.sum().item()):5d}",
                flush=True,
            )

    centroids_tensor = torch.stack(
        centroids
    )

    counts_tensor = torch.tensor(
        counts,
        dtype=torch.long,
    )

    return (
        assignments,
        centroids_tensor,
        counts_tensor,
    )


def hierarchy_step(
    vectors: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return natural_cover(
        vectors
    )


# ---------------------------------------------------------------------------
# Semantic evaluation
# ---------------------------------------------------------------------------

def semantic_coherence(
    words: list[str],
    assignments: torch.Tensor,
    semantics: dict[str, dict[str, float]],
) -> float:
    values = []

    for cluster_id in torch.unique(
        assignments
    ).tolist():
        member_indices = torch.where(
            assignments == cluster_id
        )[0].tolist()

        members = [
            words[index]
            for index in member_indices
            if words[index] in semantics
        ]

        if len(members) < 2:
            continue

        cluster_scores = []

        for i in range(
            len(members)
        ):
            for j in range(
                i + 1,
                len(members),
            ):
                cluster_scores.append(
                    weighted_jaccard(
                        semantics[members[i]],
                        semantics[members[j]],
                    )
                )

        if cluster_scores:
            values.append(
                sum(cluster_scores)
                / len(cluster_scores)
            )

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def semantic_rank_corr(
    words: list[str],
    vectors: torch.Tensor,
    semantics: dict[str, dict[str, float]],
) -> float:
    index = {
        word: i
        for i, word in enumerate(words)
    }

    anchors = [
        word
        for word in semantics
        if word in index
    ]

    if len(anchors) < 20:
        return 0.0

    x = F.normalize(
        vectors.float(),
        dim=1,
    )

    predicted = []
    target = []

    for word in anchors:
        i = index[word]

        scores = x @ x[i]

        order = torch.argsort(
            scores,
            descending=True,
        )

        used = 0

        for candidate in order:
            j = int(
                candidate.item()
            )

            if j == i:
                continue

            other = words[j]

            if other not in semantics:
                continue

            predicted.append(
                float(
                    scores[j].item()
                )
            )

            target.append(
                weighted_jaccard(
                    semantics[word],
                    semantics[other],
                )
            )

            used += 1

            if used >= SEMANTIC_NEIGHBORS:
                break

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

    p_order = torch.argsort(p)
    t_order = torch.argsort(t)

    p_rank = torch.empty_like(p)
    t_rank = torch.empty_like(t)

    p_rank[p_order] = torch.arange(
        p.numel(),
        dtype=p.dtype,
    )

    t_rank[t_order] = torch.arange(
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    print(
        "=== V105 LIVE LLM NATURAL ACTIVATION TOPOLOGY ==="
    )
    print(
        "No PT artifact."
    )
    print(
        "No target prototype count."
    )
    print(
        f"cosine_radius={COSINE_RADIUS}"
    )
    print(
        f"min_cluster_size={MIN_CLUSTER_SIZE}"
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

    raw = extract_layer3(
        tokenizer,
        model,
        device,
        words,
    )

    print()
    print(
        "activation_shape:",
        tuple(raw.shape),
    )

    # ---------------------------------------------------------------
    # Natural recursive hierarchy.
    # ---------------------------------------------------------------

    current = raw
    hierarchy = []

    for level in range(
        MAX_LEVELS
    ):
        print()
        print(
            f"=== NATURAL LEVEL {level} "
            f"input_units={current.shape[0]} ==="
        )

        (
            assignments,
            centroids,
            counts,
        ) = hierarchy_step(
            current
        )

        units = centroids.shape[0]

        singleton_count = int(
            (counts == 1).sum().item()
        )

        reused_count = units - singleton_count

        reduction = (
            1.0
            - (
                units
                / max(
                    1,
                    current.shape[0],
                )
            )
        )

        print(
            "prototype_units:",
            units,
        )

        print(
            "reused_prototypes:",
            reused_count,
        )

        print(
            "singleton_prototypes:",
            singleton_count,
        )

        print(
            "reduction:",
            reduction,
        )

        hierarchy.append(
            {
                "level": level,
                "assignments": assignments,
                "centroids": centroids,
                "counts": counts,
                "input_units": current.shape[0],
                "prototype_units": units,
                "reused_prototypes": reused_count,
                "singleton_prototypes": singleton_count,
                "reduction": reduction,
            }
        )

        # Only the actually reused prototype centroids should continue upward.
        # A hierarchy level is interesting only if the learned representation
        # genuinely contracts.
        if units >= current.shape[0]:
            print(
                "FIXED POINT: no contraction.",
            )
            break

        if reused_count == 0:
            print(
                "FIXED POINT: no reusable prototype.",
            )
            break

        if units <= 1:
            print(
                "FIXED POINT: single prototype.",
            )
            break

        current = centroids

    # ---------------------------------------------------------------
    # Semantic checks.
    # ---------------------------------------------------------------

    print()
    print(
        "=== SEMANTIC VALIDATION ==="
    )

    base_corr = semantic_rank_corr(
        words,
        raw,
        semantics,
    )

    print(
        "activation_semantic_spearman:",
        base_corr,
    )

    first = hierarchy[0]

    first_coherence = semantic_coherence(
        words,
        first["assignments"],
        semantics,
    )

    print(
        "level0_semantic_cluster_coherence:",
        first_coherence,
    )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------

    final_units = hierarchy[-1][
        "prototype_units"
    ]

    total_reduction = (
        1.0
        - (
            final_units
            / max(
                1,
                len(words),
            )
        )
    )

    print()
    print(
        "=== V105 SUMMARY ==="
    )

    print(
        "words:",
        len(words),
    )

    print(
        "final_units:",
        final_units,
    )

    print(
        "total_reduction:",
        total_reduction,
    )

    print(
        "levels:",
        len(hierarchy),
    )

    for item in hierarchy:
        print(
            f"level={item['level']:2d} "
            f"input={item['input_units']:6d} "
            f"units={item['prototype_units']:6d} "
            f"reused={item['reused_prototypes']:6d} "
            f"singletons={item['singleton_prototypes']:6d} "
            f"reduction={item['reduction']:.6f}"
        )

    print()
    print(
        "=== V105 INTERPRETATION ==="
    )

    print(
        "This hierarchy does not choose K."
    )

    print(
        "A unit exists because multiple activation observations "
        "fall inside the learned cosine radius."
    )

    print(
        "Recursion continues only while the representation contracts."
    )

    print(
        "The semantic corpus is evaluation-only."
    )

    print()

    torch.save(
        {
            "words": words,
            "raw_activations": raw,
            "hierarchy": hierarchy,
            "semantic_spearman": base_corr,
            "level0_semantic_coherence": first_coherence,
            "config": {
                "target_layer": TARGET_LAYER,
                "remove_top_components": REMOVE_TOP_COMPONENTS,
                "cosine_radius": COSINE_RADIUS,
                "min_cluster_size": MIN_CLUSTER_SIZE,
                "max_levels": MAX_LEVELS,
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
        "=== V105 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
