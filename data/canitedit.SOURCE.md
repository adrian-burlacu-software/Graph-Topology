# data/canitedit — CanItEdit, changes asked of code in words

`test.jsonl`: **CanItEdit** (`nuprl/CanItEdit`, split `test`): 105 Python
files, each with a change asked of it twice (descriptive and lazy), the file
as changed, and the tests the change must pass. Fetched through the Hugging
Face datasets-server rows API: `python -m research.v704.canitedit fetch`
(or `python -m regenerate --only canitedit`). License: MIT (Cassano et al.
2023, "Can It Edit? Evaluating the Ability of Large Language Models to
Follow Code Editing Instructions").

`projects/<name>/` each problem's file as it was, a project of its own;
`tests/<name>.py` its tests, kept out of the project; `runnable.json` which
problems can be measured here (`python -m research.v704.canitedit
projects`). Evaluation only: read as totals, never tuned on.
