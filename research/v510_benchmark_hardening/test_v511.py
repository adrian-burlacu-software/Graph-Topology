from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from planner import Planner

class M:
    topic=None
    last_assistant=""
    def facts(self): return []


def main():
    p=Planner(M(),trace=False)
    class G: name="request_information"
    c=p.choose(G(),"what color is the dog?",["The dog is red."],[],None)
    assert c.content=="The dog is red."
    assert p.last_trace["winner"]=="state"
    assert p.last_trace["candidates"]
    print("V511 instrumentation test: PASS")

if __name__=="__main__": main()
