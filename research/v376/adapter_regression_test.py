
from real_grounding import Edge

def main():
    e=Edge("dog","IsA","animal")
    assert e.weight == 1.0
    assert e.provenance == "conceptnet"
    print("V375 adapter contract test: PASS")

if __name__=="__main__":
    main()
