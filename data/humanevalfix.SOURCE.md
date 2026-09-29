# data/humanevalfix — human-written bugs in HumanEval, v696 rung 4

`js.jsonl`: the JavaScript config of **HumanEvalPack** (bigcode,
`bigcode/humanevalpack`, config `js`, split `test`): 164 HumanEval tasks,
each with a buggy solution written by people, its bug type and failure
symptom, and the task's tests (`console.assert`).

Fetched through the Hugging Face datasets-server rows API:
`python -m research.v696.bugs fetch` (or `python -m regenerate --only
humanevalfix`). License: MIT (HumanEvalPack, Muennighoff et al. 2023,
"OctoPack: Instruction Tuning Code Large Language Models").

Used as rung 4's independent held benchmark (`research/v696/bugs.py`):
each body typed by MultiPL-E's TypeScript signature for the same task
(157 of 164 have one), repaired by edits, judged by both test files.
