from __future__ import annotations

"""
V100 — SmolLM2-360M INTRODUCTORY INSTRUMENTATION

Assumes:
    ./llm/SmolLM2-360M

Purpose
-------
Validate that we can instrument a small local transformer and extract exactly
the kinds of representations needed for the Graph-Topology experiment.

This is deliberately NOT a semantic-learning experiment yet.

It runs four cheap probes over a controlled word set:

    1. TOKENIZATION
       What tokens represent each lexical item?

    2. HIDDEN-STATE EXTRACTION
       Can we retrieve embedding output + every transformer layer?

    3. LEXICAL GEOMETRY
       Do related spelling forms / words have measurable similarities at
       different layers?

    4. SEMANTIC ANCHOR GEOMETRY
       For the concepts available in semantics.csv, do embedding/layer
       similarities correlate with independently observed feature overlap?

Outputs
-------
    results/141.txt

The script prints:
    model/config summary
    tokenization examples
    hidden-state tensor shapes
    per-layer dimensionality and norm statistics
    nearest-neighbor examples
    semantic-vs-random similarity sanity check

It also writes:
    results/v100_smol_probe.npz

The NPZ contains:
    words
    token_ids
    offsets
    layer_vectors
    layer_norms

No model weights are modified.
"""

import csv
import math
import os
import time
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "llm" / "SmolLM2-360M"
DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics.csv"

OUTPUT_DIR = ROOT / "results"
PT_PATH = OUTPUT_DIR / "v100_smol_probe.pt"

# Keep this small for the introductory run.
MAX_WORDS = 1000
MAX_SEMANTIC_CONCEPTS = 150

# Hidden-state extraction batch size.
BATCH_SIZE = 16

# Layer-neighbor sanity checks.
TOP_NEIGHBORS = 5

# Reproducible word selection.
SEED = 9173


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def read_dictionary(
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

    words = sorted(words)

    if not words:
        raise RuntimeError(
            "No dictionary words loaded."
        )

    return words[:MAX_WORDS]


def read_semantic_features(
    path: Path,
) -> dict[str, set[str]]:
    """
    Small helper for the introductory semantic sanity check.

    Uses:
        basic_level_concept
        feature_name

    from the existing semantics.csv.
    """
    concepts: dict[str, set[str]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for raw in reader:
            concept = raw[
                "basic_level_concept"
            ].strip().lower()

            feature = raw[
                "feature_name"
            ].strip().lower()

            if not concept or not feature:
                continue

            concepts.setdefault(
                concept,
                set(),
            ).add(feature)

    # Only retain concepts that are in the dictionary, then cap the probe.
    dictionary = set(
        read_dictionary(
            DICTIONARY_PATH
        )
    )

    matched = [
        (
            concept,
            features,
        )
        for concept, features
        in concepts.items()
        if concept in dictionary
    ]

    matched.sort(
        key=lambda item: item[0]
    )

    return dict(
        matched[:MAX_SEMANTIC_CONCEPTS]
    )


def pad_rows(
    rows: list[list[int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(
        len(row)
        for row in rows
    )

    input_ids = torch.full(
        (
            len(rows),
            max_len,
        ),
        pad_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (
            len(rows),
            max_len,
        ),
        dtype=torch.long,
    )

    for i, row in enumerate(rows):
        input_ids[
            i,
            :len(row),
        ] = torch.tensor(
            row,
            dtype=torch.long,
        )

        attention_mask[
            i,
            :len(row),
        ] = 1

    return input_ids, attention_mask


def mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(
        -1
    ).to(hidden.dtype)

    summed = (
        hidden * mask
    ).sum(dim=1)

    counts = mask.sum(
        dim=1
    ).clamp_min(1.0)

    return summed / counts


def cosine_matrix(
    vectors: torch.Tensor,
) -> torch.Tensor:
    norms = torch.linalg.vector_norm(
        vectors,
        dim=1,
        keepdim=True,
    )

    normalized = vectors / norms.clamp_min(1e-12)

    return normalized @ normalized.transpose(0, 1)


def jaccard(
    a: set[str],
    b: set[str],
) -> float:
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 0.0

    return len(
        a & b
    ) / len(union)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model not found: {MODEL_PATH}"
        )

    device = choose_device()

    print(
        "MODEL PATH:",
        MODEL_PATH,
    )
    print(
        "DEVICE:",
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    model.eval()
    model.to(device)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Probe 1 — tokenization
# ---------------------------------------------------------------------------

def tokenization_probe(
    tokenizer,
    words: list[str],
) -> None:
    print(
        "=== V100 TOKENIZATION PROBE ==="
    )

    examples = [
        "cat",
        "car",
        "can",
        "cab",
        "capacity",
        "musculature",
        "abandon",
        "abandoned",
    ]

    # Keep only examples supported by the local tokenizer.
    examples.extend(
        words[:10]
    )

    seen = set()

    for word in examples:
        if word in seen:
            continue

        seen.add(word)

        token_ids = tokenizer.encode(
            word,
            add_special_tokens=False,
        )

        pieces = tokenizer.convert_ids_to_tokens(
            token_ids
        )

        print(
            f"{word:16s} "
            f"ids={token_ids} "
            f"pieces={pieces}"
        )

    print()


# ---------------------------------------------------------------------------
# Probe 2 — hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(
    tokenizer,
    model,
    device,
    words: list[str],
):
    """
    Returns mean-pooled vectors for each word at every available hidden-state
    level.

    Hugging Face hidden_states convention:
        hidden_states[0] = embedding output
        hidden_states[1..N] = transformer layers
    """
    all_layer_batches: list[list[np.ndarray]] = []
    token_rows: list[list[int]] = []

    # Determine layer count from a small dry run.
    dry = tokenizer(
        words[:1],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    dry_input = dry["input_ids"].to(device)
    dry_mask = dry["attention_mask"].to(device)

    dry_output = model(
        input_ids=dry_input,
        attention_mask=dry_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_count = len(
        dry_output.hidden_states
    )

    print(
        "hidden_state_levels:",
        hidden_count,
    )

    first_hidden = dry_output.hidden_states[0]

    print(
        "embedding_shape:",
        tuple(first_hidden.shape),
    )

    # Prepare per-level accumulators.
    layer_vectors = [
        []
        for _ in range(hidden_count)
    ]

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

        token_rows.extend(
            [
                row.tolist()
                for row in encoded[
                    "input_ids"
                ]
            ]
        )

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        for level, hidden in enumerate(
            output.hidden_states
        ):
            pooled = mean_pool(
                hidden,
                attention_mask,
            )

            layer_vectors[
                level
            ].append(
                pooled.detach().cpu()
            )

        if (
            start == 0
            or (
                start
                + BATCH_SIZE
                >= len(words)
            )
        ):
            print(
                "extracted:",
                min(
                    start + BATCH_SIZE,
                    len(words),
                ),
                "/",
                len(words),
                flush=True,
            )

    layer_vectors_np = [
        torch.cat(
            batches,
            dim=0,
        )
        for batches in layer_vectors
    ]

    layer_norms = [
        torch.linalg.vector_norm(
            vectors,
            dim=1,
        )
        for vectors
        in layer_vectors_np
    ]

    return (
        layer_vectors_np,
        layer_norms,
        token_rows,
    )


# ---------------------------------------------------------------------------
# Probe 3 — nearest neighbors
# ---------------------------------------------------------------------------

def nearest_neighbor_probe(
    words: list[str],
    layer_vectors: list[torch.Tensor],
) -> None:
    print(
        "=== V100 NEAREST-NEIGHBOR PROBE ==="
    )

    probes = [
        word
        for word in (
            "cat",
            "car",
            "can",
            "dog",
            "wolf",
            "fish",
            "table",
            "chair",
            "abandon",
            "abdomen",
        )
        if word in words
    ]

    indices = {
        word: index
        for index, word in enumerate(words)
    }

    for level, vectors in enumerate(
        layer_vectors
    ):
        normalized = (
            vectors
            / torch.linalg.vector_norm(
                vectors,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        similarities = (
            normalized @ normalized.transpose(0, 1)
        )

        print(
            f"--- layer {level} ---"
        )

        for probe in probes[:8]:
            i = indices[probe]

            order = torch.argsort(
                similarities[i],
                descending=True,
            )

            neighbors = []

            for j_tensor in order:
                j = int(j_tensor.item())

                if j == i:
                    continue

                neighbors.append(
                    (
                        words[j],
                        float(
                            similarities[
                                i,
                                j,
                            ].item()
                        ),
                    )
                )

                if len(neighbors) >= TOP_NEIGHBORS:
                    break

            print(
                f"{probe:12s}: "
                + ", ".join(
                    f"{word}={score:.3f}"
                    for word, score
                    in neighbors
                )
            )

        # Keep the first few and final layer readable.
        if level >= 4 and level < len(layer_vectors) - 1:
            continue

    print()


# ---------------------------------------------------------------------------
# Probe 4 — semantic feature similarity
# ---------------------------------------------------------------------------

def semantic_geometry_probe(
    words: list[str],
    layer_vectors: list[torch.Tensor],
    semantic_features: dict[str, set[str]],
) -> None:
    print(
        "=== V100 SEMANTIC GEOMETRY PROBE ==="
    )

    word_set = set(words)

    semantic_words = [
        word
        for word in semantic_features
        if word in word_set
    ]

    if len(semantic_words) < 20:
        print(
            "Not enough semantic anchors for geometry probe:",
            len(semantic_words),
        )
        print()
        return

    indices = {
        word: index
        for index, word in enumerate(words)
    }

    pairs = []

    for i in range(
        len(semantic_words)
    ):
        for j in range(
            i + 1,
            len(semantic_words),
        ):
            a = semantic_words[i]
            b = semantic_words[j]

            target = jaccard(
                semantic_features[a],
                semantic_features[b],
            )

            pairs.append(
                (a, b, target)
            )

    if not pairs:
        print("No semantic pairs.")
        print()
        return

    pair_i = torch.tensor(
        [
            indices[a]
            for a, _, _ in pairs
        ],
        dtype=torch.long,
    )

    pair_j = torch.tensor(
        [
            indices[b]
            for _, b, _ in pairs
        ],
        dtype=torch.long,
    )

    target = torch.tensor(
        [
            value
            for _, _, value in pairs
        ],
        dtype=torch.float32,
    )

    print(
        "semantic_anchor_words:",
        len(semantic_words),
    )
    print(
        "semantic_pairs:",
        len(pairs),
    )

    def rankdata(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values)
        ranks = torch.empty_like(
            values,
            dtype=torch.float32,
        )
        ranks[order] = torch.arange(
            values.numel(),
            dtype=torch.float32,
        )
        return ranks

    target_rank = rankdata(target)
    target_centered = (
        target_rank
        - target_rank.mean()
    )
    target_norm = torch.linalg.vector_norm(
        target_centered
    )

    for level, vectors in enumerate(
        layer_vectors
    ):
        normalized = (
            vectors
            / torch.linalg.vector_norm(
                vectors,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        similarities = (
            normalized[pair_i]
            * normalized[pair_j]
        ).sum(dim=1)

        pred_rank = rankdata(
            similarities
        )

        pred_centered = (
            pred_rank
            - pred_rank.mean()
        )

        pred_norm = torch.linalg.vector_norm(
            pred_centered
        )

        spearman = (
            torch.dot(
                pred_centered,
                target_centered,
            )
            / (
                target_norm
                * pred_norm
            ).clamp_min(1e-12)
        )

        print(
            f"layer={level:2d} "
            f"semantic_similarity_rank_corr={float(spearman):+.4f}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== V100 SmolLM2-360M INTRODUCTORY INSTRUMENTATION ==="
    )
    print(
        "model:",
        MODEL_PATH,
    )
    print()

    words = read_dictionary(
        DICTIONARY_PATH
    )

    semantic_features = read_semantic_features(
        SEMANTICS_PATH
    )

    print(
        "dictionary_probe_words:",
        len(words),
    )
    print(
        "semantic_anchor_words:",
        len(semantic_features),
    )
    print()

    tokenizer, model, device = load_model()

    print(
        "model_class:",
        type(model).__name__,
    )

    if hasattr(model, "config"):
        print(
            "hidden_size:",
            getattr(
                model.config,
                "hidden_size",
                "unknown",
            ),
        )
        print(
            "layers:",
            getattr(
                model.config,
                "num_hidden_layers",
                "unknown",
            ),
        )
        print(
            "vocab_size:",
            getattr(
                model.config,
                "vocab_size",
                "unknown",
            ),
        )

    print()

    tokenization_probe(
        tokenizer,
        words,
    )

    layer_vectors, layer_norms, token_rows = (
        extract_hidden_states(
            tokenizer,
            model,
            device,
            words,
        )
    )

    print(
        "=== V100 HIDDEN STATE SUMMARY ==="
    )

    for level, vectors in enumerate(
        layer_vectors
    ):
        norms = layer_norms[level]

        print(
            f"layer={level:2d} "
            f"shape={tuple(vectors.shape)} "
            f"mean_norm={float(norms.mean()):.4f} "
            f"std_norm={float(norms.std()):.4f}"
        )

    print()

    nearest_neighbor_probe(
        words,
        layer_vectors,
    )

    semantic_geometry_probe(
        words,
        layer_vectors,
        semantic_features,
    )

    torch.save(
        {
            "words": words,
            "token_ids": token_rows,
            "layer_vectors": layer_vectors,
            "layer_norms": layer_norms,
        },
        PT_PATH,
    )

    print(
        "saved:",
        PT_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - start:.2f}",
    )

    print(
        "=== V100 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
