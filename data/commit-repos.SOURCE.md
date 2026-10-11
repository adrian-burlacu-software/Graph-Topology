# Commit histories of Python projects

Whole git histories (bare clones) of 22 Python projects, read for their
commits that change several files: for each function a commit changes,
the message, the function before, and its change -- what the editor is
taught its part of a larger change from (`research/v703/teach_parts.py`).
History is read as it was before 2026-10-01 (`UNTIL`).

- Fetched by `python -m research.v703.teach_parts fetch` (`git clone
  --bare https://github.com/<owner>/<name>.git`), into
  `data/commit-repos/<owner>_<name>.git`. About 540 MB together.
- Retrieved: 2026-10-10

| Repository | Licence |
|---|---|
| pallets/flask, pallets/click, pallets/jinja, pallets/werkzeug | BSD-3-Clause |
| psf/requests | Apache-2.0 |
| encode/httpx, encode/starlette | BSD-3-Clause |
| Textualize/rich | MIT |
| python-attrs/attrs | MIT |
| fastapi/fastapi, fastapi/typer | MIT |
| psf/black | MIT |
| PyCQA/flake8 | MIT |
| pytest-dev/pytest | MIT |
| marshmallow-code/marshmallow | MIT |
| tox-dev/tox | MIT |
| pypa/hatch | MIT |
| httpie/cli | BSD-3-Clause |
| scrapy/scrapy | BSD-3-Clause |
| aio-libs/aiohttp | Apache-2.0 |
| python-poetry/poetry | MIT |
| pypa/pip | MIT |

Each project's licence is its repository's own, as published; the
clones are kept under `data/`, which is not committed.
