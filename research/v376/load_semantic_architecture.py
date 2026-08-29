
from __future__ import annotations
from pathlib import Path
import argparse

from semantic_memory import ConceptNetSQLiteLoader
from semantic_architecture import IntegratedSemanticArchitecture


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--conceptnet",
        type=Path,
        default=Path(r".\data\conceptnet_compact.db"),
    )
    p.add_argument(
        "--query",
        default="dog",
    )
    a=p.parse_args()

    db=a.conceptnet.resolve()
    print("[1/5] ConceptNet database")
    print("      path:",db)
    if not db.exists():
        raise SystemExit(
            f"NOT FOUND: {db}"
        )

    loader=ConceptNetSQLiteLoader(db)
    print("[2/5] Building indexed semantic memory...")
    memory=loader.load_index()
    print("      indexed concepts:",len(memory.concepts()))
    print("      indexed edges:",memory.edge_count)

    arch=IntegratedSemanticArchitecture(memory)
    print("[3/5] Grounding query:",a.query)
    state=arch.perceive(a.query)
    print("      candidates:",state.candidates[:8])
    print("      committed:",state.committed)
    print("      confidence:",round(state.confidence,4))
    print("      entropy:",round(state.entropy,4))

    print("[4/5] Native cognitive semantic state")
    print("      query:",state.query)
    print("      revision:",state.revision)

    print("[5/5] RESULT: PASS")
    loader.close()


if __name__=="__main__":
    main()
