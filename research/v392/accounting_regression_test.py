
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from babylm_grammar import GrammarCognitiveLearner
from semantic_architecture import IntegratedSemanticArchitecture
from semantic_memory import IndexedSemanticMemory,SemanticEdge


def main():
    memory=IndexedSemanticMemory.from_edges([
        SemanticEdge("dog","IsA","animal"),
        SemanticEdge("chases","RelatedTo","pursuit"),
        SemanticEdge("cat","IsA","animal"),
    ])
    arch=IntegratedSemanticArchitecture(memory)
    learner=GrammarCognitiveLearner(arch)

    inputs=[
        "the dog chases the cat",
        "xyz",
        "the cat chases the dog",
    ]

    for s in inputs:
        learner.observe_sentence(s,learn=True)

    assert learner.corpus_sentences_seen==3
    assert learner.grammar_observations==2
    assert learner.empty_hypothesis_sentences==1
    assert learner.grammar.memory.sentence_count==2

    print("V382 accounting seam regression: PASS")


if __name__=="__main__":
    main()
