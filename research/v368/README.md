
# V368 — Semantic Graph + BabyLM Grammar Loader

This is the first grammar loading boundary.

The loader:
1. builds/loads an explicit semantic graph;
2. loads BabyLM raw text from a local directory/file;
3. constructs an explicit `GrammarModel`;
4. validates grammar invariants;
5. runs deterministic parse probes.

The local BabyLM corpus can be supplied as:

```powershell
python .\load_grammar.py .\data\BabyLM-2026-Strict-Small
```

For a quick real-data check:

```powershell
python .\load_grammar.py .\data\BabyLM-2026-Strict-Small --limit 1000
```

The smoke test needs no dataset:

```powershell
python .\load_grammar.py
```

This first loader intentionally does not claim that heuristic category induction
is a full English grammar learner. It establishes the explicit grammar artifact
and the loading/validation interface that the later grammar learner can replace.


## Run from the project root

From the repository/project root, use:

```powershell
python .\research\v368\load_grammar.py .\data\BabyLM-2026-Strict-Small
```

Or let the loader find the canonical dataset automatically:

```powershell
python .\research\v368\load_grammar.py
```

The loader prints each path it checks and clearly reports `FOUND`, `DATASET NOT
FOUND`, `LOAD FAILED`, or `PASS`.

Data-free smoke test:

```powershell
python .\research\v368\load_grammar.py --smoke
```
