
from dataclasses import dataclass

@dataclass
class MockConceptNet:
    adj: dict
    rev: dict
    edge_count: int

class CaseAdapter:
    def __init__(self, base):
        self.adj=base.adj
        self.reverse=getattr(
            base,"reverse",
            getattr(base,"rev",None)
        )
        if self.reverse is None:
            raise AttributeError(
                "missing reverse/rev adjacency"
            )

def main():
    base=MockConceptNet({}, {}, 0)
    a=CaseAdapter(base)
    assert a.reverse is base.rev
    print("V377 adapter seam regression: PASS")

if __name__=="__main__":
    main()
