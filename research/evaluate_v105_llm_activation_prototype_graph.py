from __future__ import annotations

"""
V104B — LIVE SmolLM2 ACTIVATION -> PROTOTYPE GRAPH

This version does NOT read v102_smol_representation_geometry.pt.

It loads the local model directly from:
    ./llm/SmolLM2-360M

Pipeline
--------
ALL dictionary words
    ↓
SmolLM2-360M
    ↓
layer 3 hidden states
    ↓
mean-center across all words
    ↓
remove top common component
    ↓
512 cosine-space activation prototypes
    ↓
recursive prototype hierarchy
    ↓
semantic coherence check

This is the same representation selected by V102, but computed live.

The purpose is to validate the activation->prototype graph independently of
the saved PT artifact.

No Graph-Topology semantic labels are used to build the prototypes.
The semantic corpus is only used for evaluation after the prototype graph is
constructed.

Dependencies:
    torch
    transformers
"""

import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
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
    / "v104b_live_activation_prototype_graph.pt"
)

# Full lexical corpus.
MAX_WORDS = None

# Chosen from V102.
TARGET_LAYER = 3
REMOVE_TOP_COMPONENTS = 1

# Prototype hierarchy.
BASE_PROTOTYPES = 512
HIERARCHY_RATIO = 4
KMEANS_ITERS = 10

BATCH_SIZE = 16
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

    if MAX_WORDS is not None:
        result = result[:MAX_WORDS]

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


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def load_model():
    device = choose_device()

    print(
        "MODEL:",
        MODEL_PATH,
    )
    print(
        "DEVICE:",
        device,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model directory: {MODEL_PATH}"
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

    layers = getattr(
        model.config,
        "num_hidden_layers",
        None,
    )

    if layers is None:
        raise RuntimeError(
            "Could not determine transformer layer count."
        )

    if TARGET_LAYER > layers:
        raise RuntimeError(
            f"TARGET_LAYER={TARGET_LAYER} exceeds model depth={layers}"
        )

    print(
        "TARGET_LAYER:",
        TARGET_LAYER,
    )
    print(
        "HIDDEN_SIZE:",
        getattr(
            model.config,
            "hidden_size",
            "unknown",
        ),
    )

    return tokenizer, model, device


@torch.no_grad()
def extract_target_layer(
    tokenizer,
    model,
    device,
    words: list[str],
) -> torch.Tensor:
    """
    Extract mean-pooled TARGET_LAYER activations as float32 CPU tensors.
    """
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

    return torch.cat(
        batches,
        dim=0,
    ).float()


# ---------------------------------------------------------------------------
# V102 representation normalization
# ---------------------------------------------------------------------------

def clean_representation(
    vectors: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Reproduce the V102-selected representation:
        raw
        -> mean centered
        -> top-1 common component removed
    """
    x = vectors.float()

    mean = x.mean(
        dim=0
    )

    centered = x - mean

    work = centered.cpu()

    q = min(
        REMOVE_TOP_COMPONENTS + 4,
        work.shape[1] - 1,
        work.shape[0] - 1,
    )

    if q <= 0:
        return (
            centered,
            mean,
            torch.empty(
                (
                    0,
                    work.shape[1],
                ),
                dtype=work.dtype,
            ),
        )

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

    return (
        corrected,
        mean,
        components,
    )


# ---------------------------------------------------------------------------
# Spherical k-means
# ---------------------------------------------------------------------------

def initialize_centroids(
    vectors: torch.Tensor,
    k: int,
) -> torch.Tensor:
    x = F.normalize(
        vectors.float(),
        dim=1,
    )

    n = x.shape[0]

    if k >= n:
        return x.clone()

    selected = [0]

    distances = (
        1.0
        - (
            x @ x[0]
        )
    )

    for _ in range(
        1,
        k,
    ):
        index = int(
            torch.argmax(
                distances
            ).item()
        )

        selected.append(
            index
        )

        next_distance = (
            1.0
            - (
                x @ x[index]
            )
        )

        distances = torch.minimum(
            distances,
            next_distance,
        )

    return x[
        torch.tensor(
            selected,
            dtype=torch.long,
        )
    ]


def spherical_kmeans(
    vectors: torch.Tensor,
    k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    x = F.normalize(
        vectors.float(),
        dim=1,
    )

    k = max(
        1,
        min(
            k,
            x.shape[0],
        ),
    )

    centroids = initialize_centroids(
        x,
        k,
    )

    for iteration in range(
        KMEANS_ITERS
    ):
        similarities = (
            x @ centroids.T
        )

        assignments = torch.argmax(
            similarities,
            dim=1,
        )

        new_centroids = torch.zeros_like(
            centroids
        )

        for cluster_id in range(k):
            mask = (
                assignments
                == cluster_id
            )

            if not bool(
                mask.any()
            ):
                # Replace empty clusters with the least-covered observation.
                nearest = similarities.max(
                    dim=1
                ).values

                replacement = int(
                    torch.argmin(
                        nearest
                    ).item()
                )

                new_centroids[
                    cluster_id
                ] = x[replacement]
            else:
                mean = x[mask].mean(
                    dim=0
                )

                new_centroids[
                    cluster_id
                ] = F.normalize(
                    mean.unsqueeze(0),
                    dim=1,
                )[0]

        if torch.allclose(
            new_centroids,
            centroids,
            atol=1e-5,
            rtol=1e-5,
        ):
            centroids = new_centroids
            break

        centroids = new_centroids

        print(
            f"      kmeans {iteration + 1}/{KMEANS_ITERS}",
            flush=True,
        )

    assignments = torch.argmax(
        x @ centroids.T,
        dim=1,
    )

    return (
        assignments,
        centroids,
    )


# ---------------------------------------------------------------------------
# Semantic evaluation
# ---------------------------------------------------------------------------

def weighted_jaccard(
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


def semantic_cluster_coherence(
    words: list[str],
    assignments: torch.Tensor,
    semantics: dict[str, dict[str, float]],
) -> float:
    word_to_index = {
        word: index
        for index, word in enumerate(words)
    }

    values = []

    cluster_count = int(
        assignments.max().item()
    ) + 1

    for cluster_id in range(
        cluster_count
    ):
        member_indices = torch.where(
            assignments
            == cluster_id
        )[0].tolist()

        semantic_members = [
            words[index]
            for index in member_indices
            if words[index] in semantics
        ]

        if len(semantic_members) < 2:
            continue

        pair_values = []

        for i in range(
            len(semantic_members)
        ):
            for j in range(
                i + 1,
                len(semantic_members),
            ):
                pair_values.append(
                    weighted_jaccard(
                        semantics[
                            semantic_members[i]
                        ],
                        semantics[
                            semantic_members[j]
                        ],
                    )
                )

        if pair_values:
            values.append(
                sum(pair_values)
                / len(pair_values)
            )

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def semantic_pair_correlation(
    words: list[str],
    vectors: torch.Tensor,
    semantics: dict[str, dict[str, float]],
) -> float:
    word_to_index = {
        word: index
        for index, word in enumerate(words)
    }

    anchors = [
        word
        for word in semantics
        if word in word_to_index
    ]

    if len(anchors) < 20:
        return 0.0

    normalized = F.normalize(
        vectors.float(),
        dim=1,
    )

    predicted = []
    target = []

    for i, word in enumerate(
        anchors
    ):
        index = word_to_index[word]

        scores = (
            normalized
            @ normalized[index]
        )

        order = torch.argsort(
            scores,
            descending=True,
        )

        taken = 0

        for j_tensor in order:
            j = int(
                j_tensor.item()
            )

            if j == index:
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

            taken += 1

            if taken >= SEMANTIC_NEIGHBORS:
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
        "=== V104B LIVE SmolLM2 ACTIVATION PROTOTYPE GRAPH ==="
    )
    print(
        "No v102 PT file is used."
    )
    print(
        "Model:",
        MODEL_PATH,
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

    raw_vectors = extract_target_layer(
        tokenizer,
        model,
        device,
        words,
    )

    print()
    print(
        "RAW ACTIVATIONS:",
        tuple(raw_vectors.shape),
    )

    cleaned, mean, components = (
        clean_representation(
            raw_vectors
        )
    )

    print(
        "CLEANED ACTIVATIONS:",
        tuple(cleaned.shape),
    )

    print(
        "COMMON_COMPONENTS_REMOVED:",
        components.shape[0],
    )

    print()

    # ---------------------------------------------------------------
    # Base semantic geometry.
    # ---------------------------------------------------------------

    baseline_corr = (
        semantic_pair_correlation(
            words,
            cleaned,
            semantics,
        )
    )

    print(
        "raw_activation_semantic_spearman:",
        baseline_corr,
    )

    # ---------------------------------------------------------------
    # Prototype hierarchy.
    # ---------------------------------------------------------------

    current_vectors = cleaned

    hierarchy = []

    target_k = min(
        BASE_PROTOTYPES,
        current_vectors.shape[0],
    )

    level = 0

    while True:
        print()
        print(
            f"=== PROTOTYPE LEVEL {level} "
            f"input={current_vectors.shape[0]} "
            f"k={target_k} ==="
        )

        assignments, centroids = (
            spherical_kmeans(
                current_vectors,
                target_k,
            )
        )

        counts = torch.bincount(
            assignments,
            minlength=centroids.shape[0],
        )

        actual_k = centroids.shape[0]

        reduction = (
            1.0
            - (
                actual_k
                / max(
                    1,
                    current_vectors.shape[0],
                )
            )
        )

        coherence = (
            semantic_cluster_coherence(
                words,
                assignments,
                semantics,
            )
            if level == 0
            else 0.0
        )

        print(
            "prototype_units:",
            actual_k,
        )

        print(
            "reduction_this_level:",
            reduction,
        )

        print(
            "mean_cluster_size:",
            float(
                counts.float().mean()
            ),
        )

        print(
            "max_cluster_size:",
            int(
                counts.max().item()
            ),
        )

        if level == 0:
            print(
                "semantic_cluster_coherence:",
                coherence,
            )

        hierarchy.append(
            {
                "level": level,
                "assignments": assignments,
                "centroids": centroids,
                "counts": counts,
                "semantic_cluster_coherence": coherence,
            }
        )

        if actual_k <= 4:
            break

        next_k = max(
            4,
            actual_k // HIERARCHY_RATIO,
        )

        if next_k >= actual_k:
            break

        current_vectors = centroids
        target_k = next_k
        level += 1

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    final_units = hierarchy[-1][
        "centroids"
    ].shape[0]

    overall_reduction = (
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
        "=== V104B SUMMARY ==="
    )

    print(
        "words:",
        len(words),
    )

    print(
        "final_prototype_units:",
        final_units,
    )

    print(
        "raw_to_final_reduction:",
        overall_reduction,
    )

    print(
        "activation_semantic_spearman:",
        baseline_corr,
    )

    for item in hierarchy:
        print(
            f"level={item['level']} "
            f"units={item['centroids'].shape[0]} "
            f"coherence={item['semantic_cluster_coherence']:.6f}"
        )

    # ---------------------------------------------------------------
    # Save only AFTER successful computation.
    # ---------------------------------------------------------------

    torch.save(
        {
            "words": words,
            "cleaned_vectors": cleaned,
            "raw_vectors": raw_vectors,
            "mean": mean,
            "removed_components": components,
            "hierarchy": hierarchy,
            "activation_semantic_spearman": baseline_corr,
            "config": {
                "target_layer": TARGET_LAYER,
                "remove_top_components": REMOVE_TOP_COMPONENTS,
                "base_prototypes": BASE_PROTOTYPES,
                "hierarchy_ratio": HIERARCHY_RATIO,
                "kmeans_iterations": KMEANS_ITERS,
            },
        },
        OUTPUT_PATH,
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
        "=== V104B COMPLETE ==="
    )


if __name__ == "__main__":
    main()
