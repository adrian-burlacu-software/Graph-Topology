# NL2Bash

English descriptions of what a Bash command does, each beside the command:
12,607 pairs collected from question-and-answer sites and tutorials and
written over by Bash programmers.

- Paper: Lin, Wang, Zettlemoyer & Ernst, *NL2Bash: A Corpus and Semantic
  Parser for Natural Language Interface to the Linux Operating System*,
  LREC 2018. arXiv:1802.08979
- Home: https://github.com/TellinaTool/nl2bash
- Downloaded from:
  https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl
  and `.../all.cm` (1.6 MB together), by
  `python -m research.v702.teach_shell fetch`.
- Licence: **GNU General Public License v3.0** (the repository's). Kept
  under `data/`, which is not committed.
- Retrieved: 2026-10-09

## Files

- `nl2bash/all.nl` -- one English description per line.
- `nl2bash/all.cm` -- the command it describes, on the same line.

## Used for

`research/v702/teach_shell.py`: the shared reader's shell rows (a task, a
command given) and the Bash writer's (`llm/shell-writer`): a request, and
the command that does it. Split by the command's hash, so a command is
held out wherever it appears.
