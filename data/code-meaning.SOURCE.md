# data/code-meaning — what v696's reader of meaning is taught from

Made here, not downloaded: `python -m regenerate --only code-meaning`
(`research/v696/teach_meaning.py`). About two hours on the GPU.

| file | what | kept when |
|---|---|---|
| `docs.jsonl` | every library member of `lib.d.ts` with its JSDoc, run on generated values | always (the compiler's own words) |
| `solutions.jsonl` | MBPP-TS solved by SmolLM3-3B (4 samples) | the task's own tests pass |
| `solutions-held.jsonl` | HumanEval-TS solved the same way — **to measure with only, never taught from** | the task's own tests pass |
| `described.jsonl` | generated programs (rungs 1–2, seeds from 5,000,000) said in English by SmolLM3 | code rewritten from the English alone does what the program does |
| `corpus.jsonl` | the records: English, signature, examples, body, and the `Meaning` read exactly (behaviour by running on the tests, structure by the compiler) | — |

Sources: MultiPL-E (`data/multipl-e.SOURCE.md`), TypeScript's `lib.d.ts`
(the global `typescript` package), SmolLM3-3B (`llm/SmolLM3-3B`).

SmolLM3 samples, so a rebuild is not identical line for line; the
regenerate check asks for every source at about its size. First build,
2026-09-28: 200 docs; MBPP 257/375 solved; HumanEval 99/158 solved;
generated 1,290 of 4,000 described.
