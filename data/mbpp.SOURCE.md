# data/mbpp — MBPP, calls and what they give

`train.parquet`, `validation.parquet`: **MBPP** (Austin et al. 2021,
"Program Synthesis with Large Language Models"; Hugging Face
`google-research-datasets/mbpp`, the `full` config), CC-BY 4.0 -- Python
tasks, each with three asserts of a call and what it gives. Fetched by
`python -m research.v698.teach_examples fetch`.

Only the train and validation splits are read: their calls are said in
sentences that teach the reader where a request says an example
(`research/v698/teach_examples.py`, `llm/example-data/`). The test split is
never fetched.
