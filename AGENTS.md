# Research Agent

The goal is to evolve a developmental neural system that starts from exactly
one cell and learns a small artificial vocabulary.

The agent may modify research/genome.py only.

Do not modify:

- research/evaluate.py
- research/simulator.py
- research/config.json
- the dataset
- the scoring function

Every experiment must:

1. start from one cell;
2. run the complete developmental process;
3. report accuracy, neuron count, synapse count, reuse, and score;
4. record the mutation and result in results/.

Prefer small mutations over large rewrites.

Never claim an experiment succeeded unless it was actually run.