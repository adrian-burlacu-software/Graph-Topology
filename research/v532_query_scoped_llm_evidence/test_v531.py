from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from evidence_selector import select_evidence, compact_evidence
from query_target import infer_target


def main():
    target=infer_target("how many hands do people usually have?")
    assert target.kind=="count"
    assert target.subject=="hand"
    assert target.qualifier=="people"

    noisy=[
        {"subject":"people","predicate":"hypernym","object_text":"group","fact_type":"lexical","relevance_final":9},
        {"subject":"people","predicate":"has_property","object_text":"stupid","fact_type":"semantic","relevance_final":8},
        {"subject":"people","predicate":"capable_of","object_text":"talk","fact_type":"semantic","relevance_final":8},
        {"subject":"people","predicate":"has","object_text":"feelings","fact_type":"semantic","relevance_final":8},
        {"subject":"people","predicate":"has","object_text":"hand","fact_type":"semantic","relevance_final":8},
    ]
    selected=select_evidence(noisy,target,max_items=8,context_subject=target.qualifier)
    assert len(selected)==1, selected
    assert selected[0]["object_text"]=="hand"

    compact=compact_evidence(selected)
    assert set(compact[0])=={"subject","predicate","object","type"}
    print("V531 evidence-boundary regression: PASS")


if __name__=="__main__":
    main()
