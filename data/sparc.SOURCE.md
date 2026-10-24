# data/sparc — SParC, sequences of questions of a database

`dev.json`, `train.json`: **SParC** (Yu et al. 2019, "SParC: Cross-Domain
Semantic Parsing in Context") as mirrored by `aherntech/sparc` on Hugging
Face -- each sequence its questions and the SQL of each. License: CC BY-SA
4.0. `databases/<db_id>.sqlite`: the 20 Spider databases dev asks of, as
mirrored by `ashishkattamuri/multispider-processed` (Spider, Yu et al. 2018,
CC BY-SA 4.0). Fetched by `python -m research.v704.sparc fetch`;
`projects/<db_id>/data/<table>.csv` each database's tables (`python -m
research.v704.sparc projects`).

200 of dev's sequences said, chosen once by a seed, read as totals (those
whose last answer is empty, does not run, or has more than 10 values left
out); teaching only from train (`research/v704/sparc.py`).
