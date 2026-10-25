# data/repoqa — RepoQA, functions asked for by what they do

`repoqa-2024-06-23.json`: **RepoQA**'s Search Needle Function dataset (Liu
et al. 2024, "RepoQA: Evaluating Long Context Code Understanding"), the
`dev-dataset` release of github.com/evalplus/repoqa -- 10 repositories in
each of 6 languages, each with its code and 10 functions described in
words. License: Apache 2.0 (the repositories' code under their own
licenses). Fetched by `python -m research.v704.repoqa fetch`;
`projects/<n>/` the code given with each Python and TypeScript needle
(`python -m research.v704.repoqa projects`).

200 needles said, read as totals; nothing taught from them
(`research/v704/repoqa.py`).
