from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re, time
try:
    # Package execution:
    #   python -m v130_cognitive_architecture.integrated_eval
    from .long_term_memory import LongTermMemory
    from .modular_attention import AttentionModule
    from .working_memory import WorkingMemory
    from .designer import GraphDesigner
except ImportError:
    # Direct script execution:
    #   python .\\v130_cognitive_architecture\\integrated_eval.py
    #
    # Import the directory as a real package so that the other modules'
    # package-relative imports continue to work.
    import sys
    from pathlib import Path as _Path

    _RESEARCH_ROOT = _Path(__file__).resolve().parents[1]
    if str(_RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(_RESEARCH_ROOT))

    from v130_cognitive_architecture.long_term_memory import LongTermMemory
    from v130_cognitive_architecture.modular_attention import AttentionModule
    from v130_cognitive_architecture.working_memory import WorkingMemory
    from v130_cognitive_architecture.designer import GraphDesigner


# integrated_eval.py lives at:
#   <repo>/research/v130_cognitive_architecture/integrated_eval.py
#
# data/ and results/ live one directory above research/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data"
RESULTS_PATH = REPO_ROOT / "results"

DB_PATH = DATA_PATH / "conceptnet_compact.db"
DICTIONARY_PATH = DATA_PATH / "dictionary.csv"
OUT_PATH = RESULTS_PATH / "v130_cognitive_architecture_report.json"
LTM_PATH = RESULTS_PATH / "v130_long_term_memory.json"

@dataclass(frozen=True)
class Case:
    sentence: str
    cues: tuple[str, ...]
    expected_active: tuple[str, ...]
    bindings: tuple[tuple[str, str, str], ...] = ()
    inhibit: tuple[str, ...] = ()
    accumulate: tuple[tuple[str, str, str], ...] = ()

CASES = [
    Case("The dog is an animal.", ("dog",), ("dog","animal"), (("dog","IsA","animal"),)),
    Case("The dog can bark.", ("dog","bark"), ("dog","bark"), (("dog","CapableOf","bark"),)),
    Case("The dog has fur.", ("dog","fur"), ("dog","fur"), (("dog","HasProperty","fur"),)),
    Case("A cat is an animal.", ("cat",), ("cat","animal"), (("cat","IsA","animal"),)),
    Case("A cat can meow.", ("cat","meow"), ("cat","meow"), (("cat","CapableOf","meow"),)),
    Case("The dog chased the cat.", ("dog","chase","cat"), ("dog","chase","cat"),
         (("dog","SUBJECT","chase"),("chase","OBJECT","cat"))),
    Case("The cat chased the dog.", ("cat","chase","dog"), ("cat","chase","dog"),
         (("cat","SUBJECT","chase"),("chase","OBJECT","dog"))),
    Case("Water is a liquid.", ("water","liquid"), ("water","liquid"), (("water","IsA","liquid"),)),
    Case("Water is used for drinking.", ("water","drink"), ("water","drink"),
         (("water","UsedFor","drinking"),)),
    Case("A car is a vehicle.", ("car","vehicle"), ("car","vehicle"), (("car","IsA","vehicle"),)),
    Case("A car is used for transport.", ("car","transport"), ("car","transport"),
         (("car","UsedFor","transport"),)),
    Case("A chair is an object.", ("chair","object"), ("chair","object"),
         (("chair","IsA","object"),)),
    Case("A chair is used for sitting.", ("chair","sitting"), ("chair","sitting"),
         (("chair","UsedFor","sitting"),)),
    Case("Music is related to sound.", ("music","sound"), ("music","sound"),
         (("music","RelatedTo","sound"),)),
    Case("A bird is an animal.", ("bird",), ("bird","animal"), (("bird","IsA","animal"),)),
    Case("A bird can fly.", ("bird","fly"), ("bird","fly"), (("bird","CapableOf","fly"),)),
    Case("The bird has wings.", ("bird","wing"), ("bird","wing"), (("bird","HasA","wing"),)),
    Case("A child is a person.", ("child",), ("child","person"), (("child","IsA","person"),)),
    Case("A child can play.", ("child","play"), ("child","play"), (("child","CapableOf","play"),)),
    Case("A knife is a tool.", ("knife",), ("knife","tool"), (("knife","IsA","tool"),)),
    Case("A knife can cut.", ("knife","cut"), ("knife","cut"), (("knife","CapableOf","cut"),)),
    Case("The dog is an animal.", ("dog",), ("dog","animal"), (("dog","IsA","animal"),),
         accumulate=(("dog","IsA","animal"),)),
    Case("The dog can bark.", ("dog","bark"), ("dog","bark"), (("dog","CapableOf","bark"),),
         accumulate=(("dog","CapableOf","bark"),)),
    Case("The dog is an animal that can bark.", ("dog",), ("dog","animal","bark"),
         (("dog","IsA","animal"),("dog","CapableOf","bark")),
         accumulate=(("dog","IsA","animal"),("dog","CapableOf","bark"))),
]

def load_dictionary(path: Path) -> set[str]:
    words = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            w = raw.strip().lower()
            if w and w.isalpha():
                words.add(w)
    return words

def parse_surface_bindings(sentence: str) -> list[tuple[str,str,str]]:
    t = re.findall(r"[a-zA-Z]+", sentence.lower())
    out = []
    if "can" in t:
        i = t.index("can")
        if i > 0 and i + 1 < len(t):
            out.append((t[i-1], "CapableOf", t[i+1]))
    if "has" in t:
        i = t.index("has")
        if i > 0 and i + 1 < len(t):
            out.append((t[i-1], "HasA", t[i+1]))
    if "used" in t and "for" in t:
        i = t.index("used"); j = t.index("for")
        if i > 0 and j + 1 < len(t):
            out.append((t[i-1], "UsedFor", t[j+1]))
    if "related" in t and "to" in t:
        i = t.index("related"); j = t.index("to")
        if i > 0 and j + 1 < len(t):
            out.append((t[i-1], "RelatedTo", t[j+1]))
    # simple subject / verb / object probe
    for i, tok in enumerate(t):
        if tok in {"chased","sees","likes","helps","follows"} and i > 0 and i + 1 < len(t):
            out.append((t[i-1], "SUBJECT", tok))
            out.append((tok, "OBJECT", t[i+1]))
    return out

def run_case(
    idx: int,
    case: Case,
    memory: LongTermMemory,
    attention: AttentionModule,
    working: WorkingMemory,
    designer: GraphDesigner,
    dictionary: set[str],
) -> dict:
    attention.clear()
    working.begin(idx)

    cues = [w for w in re.findall(r"[a-zA-Z]+", case.sentence.lower()) if w in dictionary]
    for cue in cues:
        attention.seed(memory, cue, 1.0)
    for _ in range(3):
        attention.step(memory)

    active = attention.active_concepts(memory, 20)
    for concept, value in active[:8]:
        designer.reuse(concept, value)

    for source, relation, target in parse_surface_bindings(case.sentence):
        designer.bind(source, relation, target, 0.8)

    for source, relation, target in case.bindings:
        if source in memory.nodes_by_name:
            designer.reuse(source, 0.9)
        if target in memory.nodes_by_name:
            designer.reuse(target, 0.7)
        designer.bind(source, relation, target, 0.95)

    for concept in case.inhibit:
        designer.inhibit(concept)

    active_map = dict(active)
    activation_hits = {
        c: active_map.get(c, 0.0)
        for c in case.expected_active
        if active_map.get(c, 0.0) > 0
    }
    activation_coverage = len(activation_hits) / max(1, len(case.expected_active))

    binding_hits = sum(
        working.has_edge(s, r, t)
        for s, r, t in case.bindings
    )

    for s, r, t in case.accumulate:
        designer.accumulate(s, r, t, 0.05)
    designer.reinforce_semantic_edges(0.02)

    report = {
        "case": idx,
        "sentence": case.sentence,
        "cues": cues,
        "active": active,
        "activation_coverage": activation_coverage,
        "activation_hits": activation_hits,
        "binding_hits": binding_hits,
        "binding_total": len(case.bindings),
        "working_memory": working.snapshot(),
        "designer_events": [
            {"operation": e.operation, "details": e.details}
            for e in designer.events
        ],
    }

    designer.events.clear()
    working.clear()
    return report

def main() -> None:
    started = time.perf_counter()
    dictionary = load_dictionary(DICTIONARY_PATH)
    print("=== V130 WHOLE-SYSTEM COGNITIVE EVALUATION ===", flush=True)
    print("dictionary_words:", len(dictionary), flush=True)

    memory = LongTermMemory.load_conceptnet(
        DB_PATH,
        dictionary,
        min_edge_weight=1.0,
        max_edges_per_word=60,
    )
    memory.save(LTM_PATH)
    print("ltm:", memory.stats(), flush=True)

    attention = AttentionModule(
        "semantic",
        decay=0.60,
        propagation_scale=0.65,
        max_active=24,
    )
    working = WorkingMemory()
    designer = GraphDesigner(memory, working)

    episodes = []
    for idx, case in enumerate(CASES, 1):
        result = run_case(
            idx, case, memory, attention, working, designer, dictionary
        )
        episodes.append(result)
        print(
            f"CASE {idx:02d}/{len(CASES):02d} "
            f"activation={result['activation_coverage']:.3f} "
            f"bindings={result['binding_hits']}/{result['binding_total']}",
            flush=True,
        )

    total_reuse = sum(
        e["operation"] == "REUSE"
        for r in episodes for e in r["designer_events"]
    )
    total_bind = sum(
        e["operation"] == "BIND"
        for r in episodes for e in r["designer_events"]
    )
    total_accum = sum(
        e["operation"] == "ACCUMULATE"
        for r in episodes for e in r["designer_events"]
    )

    mean_activation = sum(
        r["activation_coverage"] for r in episodes
    ) / len(episodes)

    mean_binding = sum(
        r["binding_hits"] / max(1, r["binding_total"])
        for r in episodes
    ) / len(episodes)

    report = {
        "experiment": "V130 whole-system cognitive architecture",
        "episodes": len(CASES),
        "memory": memory.stats(),
        "attention": {
            "decay": attention.decay,
            "propagation_scale": attention.propagation_scale,
            "max_active": attention.max_active,
        },
        "summary": {
            "mean_activation_coverage": mean_activation,
            "mean_binding_coverage": mean_binding,
            "reuse_events": total_reuse,
            "bind_events": total_bind,
            "accumulate_events": total_accum,
        },
        "episodes_detail": episodes,
        "accumulation_probe": {
            key: (
                {
                    "reinforcement": memory.edge(*key.split("|")).reinforcement,
                    "use_count": memory.edge(*key.split("|")).use_count,
                    "effective_weight": memory.edge(*key.split("|")).effective_weight,
                }
                if memory.edge(*key.split("|")) is not None
                else None
            )
            for key in (
                "dog|IsA|animal",
                "dog|CapableOf|bark",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== V130 SUMMARY ===")
    print("mean_activation_coverage:", mean_activation)
    print("mean_binding_coverage:", mean_binding)
    print("reuse_events:", total_reuse)
    print("bind_events:", total_bind)
    print("accumulate_events:", total_accum)
    print("saved:", OUT_PATH)
    print("saved_ltm:", LTM_PATH)
    print("elapsed_seconds:", f"{time.perf_counter() - started:.2f}")
    print("=== V130 COMPLETE ===")

if __name__ == "__main__":
    main()
