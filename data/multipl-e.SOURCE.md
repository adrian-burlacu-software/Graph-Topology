# MultiPL-E (TypeScript)

HumanEval and MBPP translated to other languages, with each language's
tests. v696 uses the TypeScript configs as its independent code benchmark.

- Paper: Cassano et al., *MultiPL-E: A Scalable and Polyglot Approach to
  Benchmarking Neural Code Generation*, IEEE TSE 2023. arXiv:2208.08227
- Downloaded from: the Hugging Face datasets server, dataset
  `nuprl/MultiPL-E`, configs `humaneval-ts` (159) and `mbpp-ts` (390),
  split `test` (`python -m research.v696.tasks fetch`).
- Licence: **not checked** at download -- see the dataset card before any
  use beyond research. Kept under `data/`, which is not committed.
- Retrieved: 2026-09-27

## Use

HumanEval-TS prompts carry examples in their comments (151 readable); the
search sees those and is judged by the task's own tests. MBPP-TS prompts
carry none: its first test is shown and the rest judge.
