# data/cruxeval — CRUXEval, what code gives and what gives it

`test.jsonl`: **CRUXEval** (`cruxeval-org/cruxeval`, split `test`): 800 short
Python functions `f`, each with an input and the output it gives. Fetched
through the Hugging Face datasets-server rows API: `python -m
research.v704.cruxeval fetch` (or `python -m regenerate --only cruxeval`).
License: MIT (Gu et al. 2024, "CRUXEval: A Benchmark for Code Reasoning,
Understanding and Execution").

Evaluation only (`research/v704/cruxeval.py`): output prediction and input
prediction as conversations of pasted code; read as totals.
