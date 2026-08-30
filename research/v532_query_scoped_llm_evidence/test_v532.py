from evidence_selector import select_llm_evidence, compact_evidence
from query_target import QueryTarget

def main():
    facts = [
        {"subject":"people","predicate":"hypernym","object_text":"group","fact_type":"lexical","datasets":"wordnet","relevance_final":9},
        {"subject":"people","predicate":"has_property","object_text":"stupid","fact_type":"semantic","datasets":"conceptnet","relevance_final":8},
        {"subject":"hand","predicate":"count","object_text":"2","fact_type":"semantic","datasets":"conceptnet","relevance_final":7},
    ]
    target=QueryTarget(kind="count",subject="hand",qualifier="people",plural=True,explicit=True)
    selected=select_llm_evidence(facts,target)
    assert len(selected)==1 and selected[0]["subject"]=="hand"
    payload=compact_evidence(selected,target)
    assert payload==["The number of hand is 2."]
    assert "wordnet" not in str(payload)
    assert "relevance" not in str(payload)
    print("V532 evidence hygiene: PASS")

if __name__=="__main__":
    main()
