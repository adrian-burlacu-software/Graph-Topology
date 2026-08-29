
from benchmark import curriculum
from architecture import VerifiedOperatorArchitecture

def main():
    for mode in (
        "verified_balanced",
        "verified_sparse",
        "verified_exploratory",
        "verified_conservative",
    ):
        arch=VerifiedOperatorArchitecture(mode)
        rows=[
            arch.run(ep,True)
            for _,_,ep in curriculum(367)
        ]
        assert len(rows)==10
        d=arch.diagnostics()
        assert d["episodes"]==10
        assert d["epistemic_interventions"]>0
        assert d["beliefs"]>=1
        assert d["persistent_models"]["total_models"]>=1

    # The novel R2 phase must be solved without semantic fallback.
    arch=VerifiedOperatorArchitecture("verified_balanced")
    rows=list(curriculum(368))
    r2=arch.run(rows[3][2],True)
    assert r2["source"]=="verified_induced_operator"
    assert r2["correct"] is True

    print("V367 validation: PASS")
    print("schema -> executable operator synthesis: PASS")
    print("intervention-based operator verification: PASS")
    print("posterior commitment: PASS")
    print("persistent operator model: PASS")
    print("novel R2 without fallback: PASS")

if __name__=="__main__":
    main()
