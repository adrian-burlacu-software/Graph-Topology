# data/wtq — WikiTableQuestions, questions of tables

`test.jsonl`, `train.jsonl`: **WikiTableQuestions** as mirrored by
`lighteval/wikitablequestions` (its `test` 18 486 questions, its `train`
10 -- not the original splits), each question with its answers and its
table. Fetched through the Hugging Face datasets-server rows API: `python
-m research.v704.wtq fetch`; `projects/<id>/data/table.csv` each asked
table (`python -m research.v704.wtq projects`). License: CC BY-SA 4.0
(Pasupat & Liang 2015, "Compositional Semantic Parsing on Semi-Structured
Tables").

200 questions said, chosen once by a seed, read as totals; teaching only
from the rest, never a question said (`research/v704/wtq.py`).
