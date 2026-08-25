from __future__ import annotations

"""
V102 — SmolLM2-360M REPRESENTATION GEOMETRY STUDY

This is the full representation-selection experiment before we feed LLM
activations into Graph-Topology.

Question
--------
The raw residual-stream vectors showed almost-unit cosine similarity after
the middle layers. Is the information actually missing, or is it buried under
a large shared component?

We evaluate several representations from EVERY hidden layer:

    RAW
        mean-pooled hidden state

    CENTERED
        raw - global mean vector

    CENTERED + TOP-PC REMOVAL
        centered representation with the dominant common direction removed

    L2-NORMALIZED versions of the above

For every representation/layer we measure:

    * pairwise cosine separation
    * mean off-diagonal cosine
    * cosine std
    * nearest-neighbor diversity
    * semantic feature-overlap rank correlation
    * effective dimensionality
    * tokenization statistics

The semantic probe uses the existing human-feature `semantics.csv`.

The dictionary side uses ALL words in dictionary.csv.

This script is intentionally self-contained and NumPy-free.
It saves the corrected representation candidates to:

    results/v102_smol_representation_geometry.pt

The goal is to choose ONE representation for the next graph experiment,
not to run another sequence of tiny model variants.
"""

import csv
import hashlib
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "llm" / "SmolLM2-360M"
DICTIONARY_PATH = ROOT / "data" / "dictionary.csv"
SEMANTICS_PATH = ROOT / "data" / "semantics.csv"

OUTPUT_DIR = ROOT / "results"
SAVE_PATH = OUTPUT_DIR / "v102_smol_representation_geometry.pt"

BATCH_SIZE = 16

# Use the entire lexical dictionary.
MAX_WORDS = None

# Maximum number of semantic anchor concepts for the geometry probe.
# The current semantics.csv is small; use every matched concept available.
MAX_SEMANTIC_CONCEPTS = None

# Remove this many common principal directions after centering.
TOP_COMPONENTS = 1

# Effective rank threshold.
EIGEN_THRESHOLD = 0.99

# Nearest-neighbor sanity check.
TOP_NEIGHBORS = 8


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dictionary(
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
            "dictionary.csv produced no usable words."
        )

    return result


def load_semantic_features(
    path: Path,
) -> dict[str, set[str]]:
    concepts: dict[str, set[str]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "basic_level_concept",
            "feature_name",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "semantics.csv missing: "
                + ", ".join(sorted(missing))
            )

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

    return concepts


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def mean_center(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    return x - mean, mean


def remove_top_components(
    x: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove the dominant PCA directions from an already centered matrix.

    Uses torch.pca_lowrank so no NumPy / sklearn dependency is required.
    Returns:
        corrected matrix
        principal components
    """
    if n_components <= 0:
        return x, torch.empty(
            (0, x.shape[1]),
            dtype=x.dtype,
        )

    q = min(
        n_components + 4,
        x.shape[1] - 1,
        x.shape[0] - 1,
    )

    if q <= 0:
        return x, torch.empty(
            (0, x.shape[1]),
            dtype=x.dtype,
        )

    # PCA is done on CPU float32 for predictable memory use.
    work = x.detach().float().cpu()

    # q-approximation is more than enough for top-1/top-few common directions.
    _, _, v = torch.pca_lowrank(
        work,
        q=q,
        center=False,
    )

    components = v[:, :n_components].transpose(
        0,
        1,
    )

    corrected = (
        work
        - (
            work @ components.transpose(0, 1)
        )
        @ components
    )

    return (
        corrected,
        components,
    )


def l2_normalize(
    x: torch.Tensor,
) -> torch.Tensor:
    return x / torch.linalg.vector_norm(
        x,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)


def cosine_stats(
    x: torch.Tensor,
) -> dict[str, float]:
    normalized = l2_normalize(
        x.float()
    )

    # For 4,925 x 960 this is manageable, but avoid materializing the whole
    # similarity matrix unless it is useful. Compute blockwise off-diagonal
    # statistics instead.
    n = normalized.shape[0]
    block = 512

    total = 0
    sum_cos = 0.0
    sum_sq = 0.0
    diag_sum = 0.0

    for start in range(
        0,
        n,
        block,
    ):
        a = normalized[
            start:start + block
        ]

        sim = a @ normalized.transpose(
            0,
            1,
        )

        rows = sim.shape[0]

        row_indices = torch.arange(
            rows
        )

        global_indices = (
            torch.arange(
                start,
                start + rows,
            )
        )

        sim[
            row_indices,
            global_indices,
        ] = 0.0

        values = sim.reshape(-1)

        # The diagonal has been zeroed, so remove those zeros from stats.
        values = torch.cat(
            [
                values[:0],
                values[values != 0.0],
            ]
        )

        count = values.numel()

        if count:
            total += count
            sum_cos += float(
                values.sum().item()
            )
            sum_sq += float(
                (values * values).sum().item()
            )

    mean = (
        sum_cos
        / max(
            1,
            total,
        )
    )

    variance = max(
        0.0,
        (
            sum_sq
            / max(1, total)
        )
        - mean * mean,
    )

    return {
        "mean_offdiag_cosine": mean,
        "std_offdiag_cosine": math.sqrt(
            variance
        ),
        "pair_count": float(total),
    }


def effective_dimensionality(
    x: torch.Tensor,
) -> float:
    """
    Participation-ratio effective dimensionality.

        (sum eigenvalues)^2 / sum eigenvalues^2
    """
    work = x.float().cpu()

    # Covariance over dimensions, not words.
    covariance = (
        work.transpose(0, 1)
        @ work
    )

    covariance /= max(
        1,
        work.shape[0] - 1,
    )

    eigenvalues = torch.linalg.eigvalsh(
        covariance
    ).clamp_min(0.0)

    total = eigenvalues.sum().item()

    if total <= 0.0:
        return 0.0

    return (
        total * total
        / max(
            1e-12,
            float(
                (eigenvalues * eigenvalues)
                .sum()
                .item()
            ),
        )
    )


def rankdata(
    values: torch.Tensor,
) -> torch.Tensor:
    order = torch.argsort(
        values
    )

    ranks = torch.empty_like(
        values,
        dtype=torch.float32,
    )

    ranks[order] = torch.arange(
        values.numel(),
        dtype=torch.float32,
    )

    return ranks


def spearman(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    ar = rankdata(a)
    br = rankdata(b)

    ar -= ar.mean()
    br -= br.mean()

    denominator = (
        torch.linalg.vector_norm(ar)
        * torch.linalg.vector_norm(br)
    ).clamp_min(
        1e-12
    )

    return float(
        torch.dot(
            ar,
            br,
        ).item()
        / denominator.item()
    )


# ---------------------------------------------------------------------------
# Semantic geometry
# ---------------------------------------------------------------------------

def build_semantic_pairs(
    words: list[str],
    semantic_features: dict[str, set[str]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    word_set = set(words)

    anchors = [
        word
        for word in semantic_features
        if word in word_set
    ]

    anchors.sort()

    if (
        MAX_SEMANTIC_CONCEPTS is not None
    ):
        anchors = anchors[
            :MAX_SEMANTIC_CONCEPTS
        ]

    index = {
        word: i
        for i, word in enumerate(words)
    }

    pair_i = []
    pair_j = []
    target = []

    for i in range(
        len(anchors)
    ):
        for j in range(
            i + 1,
            len(anchors),
        ):
            a = anchors[i]
            b = anchors[j]

            fa = semantic_features[a]
            fb = semantic_features[b]

            union = fa | fb

            if not union:
                similarity = 0.0
            else:
                similarity = (
                    len(fa & fb)
                    / len(union)
                )

            pair_i.append(
                index[a]
            )
            pair_j.append(
                index[b]
            )
            target.append(
                similarity
            )

    return (
        torch.tensor(
            pair_i,
            dtype=torch.long,
        ),
        torch.tensor(
            pair_j,
            dtype=torch.long,
        ),
        torch.tensor(
            target,
            dtype=torch.float32,
        ),
        anchors,
    )


def semantic_similarity_correlation(
    vectors: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    target: torch.Tensor,
) -> float:
    normalized = l2_normalize(
        vectors.float()
    )

    predicted = (
        normalized[pair_i]
        * normalized[pair_j]
    ).sum(dim=1)

    return spearman(
        predicted,
        target,
    )


# ---------------------------------------------------------------------------
# Tokenization / model
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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    device = choose_device()

    print(
        "MODEL:",
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
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad nor eos token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return (
        tokenizer,
        model,
        device,
    )


@torch.no_grad()
def extract_word_representations(
    tokenizer,
    model,
    device,
    words: list[str],
) -> tuple[
    list[torch.Tensor],
    list[list[int]],
]:
    """
    Extract mean-pooled hidden states for every hidden-state level.

    hidden_states[0] is the embedding output.
    hidden_states[1..N] are transformer-layer outputs.
    """
    # Dry run to discover layer count.
    encoded = tokenizer(
        words[:1],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    output = model(
        input_ids=encoded[
            "input_ids"
        ].to(device),
        attention_mask=encoded[
            "attention_mask"
        ].to(device),
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_count = len(
        output.hidden_states
    )

    print(
        "HIDDEN STATE LEVELS:",
        hidden_count,
    )

    layer_batches = [
        []
        for _ in range(hidden_count)
    ]

    token_rows = []

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

        input_ids = batch[
            "input_ids"
        ]

        attention_mask = batch[
            "attention_mask"
        ]

        token_rows.extend(
            [
                row.tolist()
                for row in input_ids
            ]
        )

        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        for level, hidden in enumerate(
            output.hidden_states
        ):
            hidden = hidden.float()

            pooled = (
                hidden * mask.to(
                    hidden.device
                )
            ).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(
                1.0
            ).to(
                hidden.device
            )

            layer_batches[
                level
            ].append(
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

    layer_vectors = [
        torch.cat(
            batches,
            dim=0,
        ).float()
        for batches in layer_batches
    ]

    return (
        layer_vectors,
        token_rows,
    )


# ---------------------------------------------------------------------------
# Nearest-neighbor sanity
# ---------------------------------------------------------------------------

def nearest_neighbors(
    vectors: torch.Tensor,
    words: list[str],
    probe_words: list[str],
) -> dict[str, list[tuple[str, float]]]:
    normalized = l2_normalize(
        vectors.float()
    )

    index = {
        word: i
        for i, word in enumerate(words)
    }

    result = {}

    for word in probe_words:
        if word not in index:
            continue

        i = index[word]

        scores = (
            normalized
            @ normalized[i]
        )

        order = torch.argsort(
            scores,
            descending=True,
        )

        neighbors = []

        for j_tensor in order:
            j = int(
                j_tensor.item()
            )

            if j == i:
                continue

            neighbors.append(
                (
                    words[j],
                    float(
                        scores[j].item()
                    ),
                )
            )

            if len(neighbors) >= TOP_NEIGHBORS:
                break

        result[word] = neighbors

    return result


def print_neighbors(
    label: str,
    vectors: torch.Tensor,
    words: list[str],
) -> None:
    probes = [
        "cat",
        "dog",
        "wolf",
        "fish",
        "car",
        "train",
        "table",
        "chair",
        "abandon",
        "abdomen",
    ]

    print(
        f"=== {label} NEAREST NEIGHBORS ==="
    )

    for word, neighbors in nearest_neighbors(
        vectors,
        words,
        probes,
    ).items():
        print(
            f"{word:12s}: "
            + ", ".join(
                f"{n}={s:.4f}"
                for n, s in neighbors
            )
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== V102 SmolLM2-360M REPRESENTATION GEOMETRY ==="
    )
    print(
        "Goal: identify a trustworthy representation before graph integration."
    )
    print()

    words = load_dictionary(
        DICTIONARY_PATH
    )

    semantic_features = load_semantic_features(
        SEMANTICS_PATH
    )

    (
        pair_i,
        pair_j,
        semantic_target,
        anchors,
    ) = build_semantic_pairs(
        words,
        semantic_features,
    )

    print(
        "dictionary_words:",
        len(words),
    )

    print(
        "semantic_anchors:",
        len(anchors),
    )

    print(
        "semantic_pairs:",
        len(
            semantic_target
        ),
    )

    print()

    tokenizer, model, device = load_model()

    print(
        "MODEL_CLASS:",
        type(model).__name__,
    )

    print(
        "HIDDEN_SIZE:",
        getattr(
            model.config,
            "hidden_size",
            "unknown",
        ),
    )

    print(
        "LAYERS:",
        getattr(
            model.config,
            "num_hidden_layers",
            "unknown",
        ),
    )

    print()

    # Tokenization summary.
    token_counts = []

    for word in words:
        ids = tokenizer.encode(
            word,
            add_special_tokens=False,
        )

        token_counts.append(
            len(ids)
        )

    print(
        "=== TOKENIZATION ==="
    )

    print(
        "mean_tokens_per_word:",
        sum(token_counts)
        / max(
            1,
            len(token_counts),
        ),
    )

    print(
        "max_tokens_per_word:",
        max(
            token_counts,
            default=0,
        ),
    )

    for word in (
        "cat",
        "dog",
        "musculature",
        "abandon",
        "abandoned",
    ):
        ids = tokenizer.encode(
            word,
            add_special_tokens=False,
        )

        print(
            word,
            "->",
            tokenizer.convert_ids_to_tokens(
                ids
            ),
        )

    print()

    layer_vectors, token_rows = (
        extract_word_representations(
            tokenizer,
            model,
            device,
            words,
        )
    )

    # ------------------------------------------------------------------
    # Evaluate every layer under every normalization.
    # ------------------------------------------------------------------

    records = []

    for level, raw in enumerate(
        layer_vectors
    ):
        raw = raw.float()

        centered, mean = mean_center(
            raw
        )

        corrected, components = (
            remove_top_components(
                centered,
                TOP_COMPONENTS,
            )
        )

        representations = {
            "raw": raw,
            "centered": centered,
            "centered_top1_removed": corrected,
        }

        for name, representation in (
            representations.items()
        ):
            # Raw geometry.
            stats = cosine_stats(
                representation
            )

            dim = effective_dimensionality(
                representation
            )

            semantic_corr = (
                semantic_similarity_correlation(
                    representation,
                    pair_i,
                    pair_j,
                    semantic_target,
                )
            )

            normalized = l2_normalize(
                representation
            )

            normalized_stats = cosine_stats(
                normalized
            )

            records.append(
                {
                    "level": level,
                    "representation": name,
                    "mean_norm": float(
                        torch.linalg.vector_norm(
                            representation,
                            dim=1,
                        ).mean().item()
                    ),
                    "std_norm": float(
                        torch.linalg.vector_norm(
                            representation,
                            dim=1,
                        ).std().item()
                    ),
                    "mean_offdiag_cosine": stats[
                        "mean_offdiag_cosine"
                    ],
                    "std_offdiag_cosine": stats[
                        "std_offdiag_cosine"
                    ],
                    "normalized_mean_offdiag_cosine": normalized_stats[
                        "mean_offdiag_cosine"
                    ],
                    "normalized_std_offdiag_cosine": normalized_stats[
                        "std_offdiag_cosine"
                    ],
                    "effective_dimensionality": dim,
                    "semantic_rank_corr": semantic_corr,
                    "center_mean_norm": float(
                        torch.linalg.vector_norm(
                            mean
                        ).item()
                    ),
                    "top_component_count": int(
                        components.shape[0]
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Print compact table.
    # ------------------------------------------------------------------

    print(
        "=== V102 REPRESENTATION SUMMARY ==="
    )

    print(
        "level | representation          | "
        "mean_cos | cos_std | eff_dim | semantic_corr | mean_norm"
    )

    for record in records:
        if (
            record["level"] in {
                0,
                1,
                2,
                4,
                len(layer_vectors) // 2,
                len(layer_vectors) - 1,
            }
        ):
            print(
                f"{record['level']:5d} | "
                f"{record['representation']:23s} | "
                f"{record['mean_offdiag_cosine']:+.6f} | "
                f"{record['std_offdiag_cosine']:.6f} | "
                f"{record['effective_dimensionality']:7.1f} | "
                f"{record['semantic_rank_corr']:+.4f} | "
                f"{record['mean_norm']:.3f}"
            )

    print()

    # ------------------------------------------------------------------
    # Select the best representation automatically by semantic rank
    # correlation, with separation as a secondary criterion.
    # ------------------------------------------------------------------

    ranked = sorted(
        records,
        key=lambda record: (
            record["semantic_rank_corr"],
            record["std_offdiag_cosine"],
            record["effective_dimensionality"],
        ),
        reverse=True,
    )

    best = ranked[0]

    print(
        "=== V102 BEST CANDIDATE ==="
    )

    for key, value in best.items():
        print(
            f"{key}: {value}"
        )

    print()

    # ------------------------------------------------------------------
    # Neighbor comparison for the selected representation.
    # ------------------------------------------------------------------

    best_level = int(
        best["level"]
    )

    if best["representation"] == "raw":
        best_vectors = layer_vectors[
            best_level
        ]
    else:
        centered, _ = mean_center(
            layer_vectors[
                best_level
            ].float()
        )

        if (
            best["representation"]
            == "centered"
        ):
            best_vectors = centered
        else:
            best_vectors, _ = (
                remove_top_components(
                    centered,
                    TOP_COMPONENTS,
                )
            )

    print_neighbors(
        "V102 BEST",
        best_vectors,
        words,
    )

    # ------------------------------------------------------------------
    # Save all useful representations for downstream graph experiments.
    # ------------------------------------------------------------------

    tensors = {}

    for level, raw in enumerate(
        layer_vectors
    ):
        centered, mean = mean_center(
            raw
        )

        corrected, components = (
            remove_top_components(
                centered,
                TOP_COMPONENTS,
            )
        )

        tensors[
            f"layer_{level}_raw"
        ] = raw

        tensors[
            f"layer_{level}_centered"
        ] = centered

        tensors[
            f"layer_{level}_centered_top1_removed"
        ] = corrected

        tensors[
            f"layer_{level}_mean"
        ] = mean

        tensors[
            f"layer_{level}_top_components"
        ] = components

    torch.save(
        {
            "words": words,
            "token_ids": token_rows,
            "semantic_anchors": anchors,
            "records": records,
            "best": best,
            "layers": tensors,
        },
        SAVE_PATH,
    )

    print(
        "saved:",
        SAVE_PATH,
    )

    print(
        "elapsed_seconds:",
        f"{time.perf_counter() - started:.2f}",
    )

    print(
        "=== V102 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
