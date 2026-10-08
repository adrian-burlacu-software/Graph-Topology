"""Rebuild everything the repository does not carry, with one command.

    python -m regenerate                 # everything missing, in order
    python -m regenerate --list          # what there is, and what is present
    python -m regenerate --only awa2 babi
    python -m regenerate --force reader  # rebuild even if it looks present
    python -m regenerate --dry-run

Nothing under `data/`, `llm/` or `state/` is committed: between them they are
tens of gigabytes of downloaded corpora, trained checkpoints and taught
memories. On 2026-09-16 all of `llm/` and 21 directories of `data/` were
destroyed at once, and what made recovery possible was not a backup but the
`SOURCE.md` beside each dataset. What made it *painful* was that the trained
models and two of the corpora had no such path, and the pipeline that built
the shipped reader existed only in a shell script that was never committed.

So: every artefact this system needs is a `Step` below, and every step says
how to make it, how to tell whether it is already there, and what it must
contain when it is. A step that cannot verify its own result is not done.

**Order matters, and it is a cycle.** The reader is read by every layer, and
without `llm/reader` nothing parses at all (`research/encoder.py`). But the
reader shipped in v690 was taught partly on the decoder's filtered replies,
and those replies are played through a running engine -- which needs a
reader. That is a two-pass cycle and it is written out as one:

    reader-corpus -> reader-first -> turns -> replies -> label
                  -> decoder -> reader        (taught on the replies too)

`reader-first` is a working reader taught only by the rules (the
`RELATION_CUES` table, via `language.taught_by_cues`); `reader` is the
shipped one. Skipping straight to `reader` is not possible from empty.

**Downloads are pinned to files, never to project pages.** Two traps found
the hard way: osf.io needs a trailing slash or answers 308, and NEWTON is on
`master` while COMPS is on `main` with its pair file one directory deeper
than the rest.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LLM = ROOT / "llm"
STATE = ROOT / "state"

#: Where nltk keeps its corpora. VerbNet 3.3 is already there as `verbnet3`,
#: so it is copied rather than downloaded -- and WordNet lives here too,
#: outside the repository, which is why the taxonomy survived the deletion.
def _nltk_corpora() -> Path | None:
    try:
        import nltk
    except ImportError:
        return None
    for base in nltk.data.path:
        found = Path(base) / "corpora"
        if found.is_dir():
            return found
    return None


class Failed(Exception):
    """A step ran but its result is not what it must be."""


@dataclass
class Step:
    """One artefact: how to make it, and how to know it is really there.

    `check` returns a short description when the artefact is present and
    correct, and raises `Failed` when it is present but wrong. Returning
    None means "not there yet". A step whose `check` only tests existence
    is a step that will one day report success over an empty file --
    `corpora.load_lvis_categories` swallows a missing file and returns `{}`,
    which is how NEWTON's join silently fell to nothing.
    """

    name: str
    what: str
    make: Callable[[], None]
    check: Callable[[], str | None]
    needs: tuple[str, ...] = ()
    #: minutes, very roughly, to say what a run is about to cost
    cost: str = ""
    gpu: bool = False


# ---------------------------------------------------------------- fetching

def _get(url: str, into: Path, note: str = "") -> Path:
    into.parent.mkdir(parents=True, exist_ok=True)
    print(f"    fetch {note or url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(request, timeout=900) as response:
        into.write_bytes(response.read())
    return into


def _lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _count(loader: Callable[[], object], expected: int, name: str) -> str:
    """Run a loader and insist on the documented count."""
    got = len(loader())                                  # type: ignore[arg-type]
    if got != expected:
        raise Failed(f"{name}: {got}, documented {expected}")
    return f"{got} {name}"


# ------------------------------------------------------------------ steps

def _awa2_make() -> None:
    archive = _get("https://cvml.ista.ac.at/AwA2/AwA2-base.zip",
                   DATA / "awa2" / "AwA2-base.zip", "AwA2-base.zip (32 KB)")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(DATA / "awa2")
    archive.unlink()


def _awa2_check() -> str | None:
    if not (DATA / "awa2" / "Animals_with_Attributes2" / "classes.txt").exists():
        return None
    from research.v687 import corpora
    return _count(lambda: corpora.load_awa2().items, 50, "classes")


def _xcslb_make() -> None:
    base = "https://raw.githubusercontent.com/kanishkamisra/comps/main/data/"
    # The pair file is a directory deeper than the other three.
    for name, where in (("comps_base.jsonl", "comps/comps_base.jsonl"),
                        ("feature_lexicon.csv", "feature_lexicon.csv"),
                        ("concept_senses.csv", "concept_senses.csv"),
                        ("concept_matrix.txt", "concept_matrix.txt")):
        _get(base + where, DATA / "xcslb" / name, name)


def _entailmentbank_make() -> None:
    """EntailmentBank, which is published on Google Drive and nowhere else.

    The folder holds three dated versions and the id below is v3's zip
    (`v3_May6_2022`), taken from the folder listing rather than guessed. A
    public Drive file downloads from `uc?export=download` without a token
    while it is under the virus-scan threshold, and at 7.8 MB this is; if
    Drive ever starts interposing the scan page, the download returns HTML
    and `_entailmentbank_check` fails on the tree count rather than
    quietly leaving a broken file.
    """
    import zipfile

    archive = _get("https://drive.google.com/uc?export=download&id="
                   "1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh",
                   DATA / "entailmentbank" / "entailmentbank.zip",
                   "entailment_trees_emnlp2021_data_v3.zip (7.8 MB)")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(DATA / "entailmentbank")
    archive.unlink()


def _entailmentbank_check() -> str | None:
    from research.v690 import entailment
    if not entailment.DATASET.exists():
        return None
    return _count(lambda: entailment.load("task_1", "dev"), 187,
                  "task 1 dev trees")


def _xcslb_check() -> str | None:
    if not (DATA / "xcslb" / "comps_base.jsonl").exists():
        return None
    from research.v687 import corpora
    return _count(lambda: corpora.load_xcslb().items, 521, "concepts")


#: THINGSplus, one osf.io id per file. The trailing slash is required:
#: without it osf.io answers 308 and nothing downloads.
THINGSPLUS = {"concepts-metadata_things.tsv": "5uvc2",
              "property-ratings.tsv": "7cz69",
              "category53_long-format.tsv": "vehr3",
              "typicality53_mean-ratings.tsv": "qj7ec",
              "object-level_description.txt": "kdyq6",
              "category-level_description.txt": "bpxye"}


def _thingsplus_make() -> None:
    for name, osf in THINGSPLUS.items():
        _get(f"https://osf.io/download/{osf}/", DATA / "thingsplus" / name, name)


def _thingsplus_check() -> str | None:
    if not (DATA / "thingsplus" / "concepts-metadata_things.tsv").exists():
        return None
    from research.v687 import corpora
    return _count(corpora.load_thingsplus, 1854, "objects")


def _newton_make() -> None:
    # NEWTON is on `master`, not `main`.
    _get("https://raw.githubusercontent.com/NewtonReasoning/Newton/master/"
         "data/confident_questions.csv",
         DATA / "newton" / "confident_questions.csv", "confident_questions.csv")
    # LVIS's categories are a Python literal in detectron2, kept here as the
    # JSON the loader reads.
    import ast
    source = urllib.request.urlopen(
        urllib.request.Request(
            "https://raw.githubusercontent.com/facebookresearch/detectron2/"
            "main/detectron2/data/datasets/lvis_v1_categories.py",
            headers={"User-Agent": "curl/8"}), timeout=300).read().decode("utf-8")
    rows = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                getattr(target, "id", "") == "LVIS_CATEGORIES"
                for target in node.targets):
            rows = ast.literal_eval(node.value)
            break
    if rows is None:
        raise Failed("LVIS_CATEGORIES not found in detectron2")
    (DATA / "newton" / "lvis_v1_categories.json").write_text(
        json.dumps(rows), encoding="utf-8")


def _newton_check() -> str | None:
    categories = DATA / "newton" / "lvis_v1_categories.json"
    if not (DATA / "newton" / "confident_questions.csv").exists():
        return None
    from research.v687 import corpora
    note = _count(corpora.load_newton, 777, "objects")
    if not categories.exists():
        raise Failed("lvis_v1_categories.json missing: load_lvis_categories "
                     "returns {} for it and NEWTON's join silently empties")
    # 1203 *rows*; the loader returns a dict keyed by name AND synonym, so
    # its length is larger and is not the number to assert.
    rows = json.loads(categories.read_text(encoding="utf-8"))
    if len(rows) != 1203:
        raise Failed(f"lvis categories: {len(rows)}, documented 1203")
    return f"{note}, 1203 lvis categories"


def _buchanan_make() -> None:
    # Upstream the name has spaces, in a directory that also has one.
    _get("https://raw.githubusercontent.com/doomlab/Word-Norms-2/master/"
         "3%20parsed/top%20to%20final.csv",
         DATA / "buchanan" / "top_to_final.csv", "top to final.csv (1.4 MB)")


def _buchanan_check() -> str | None:
    if not (DATA / "buchanan" / "top_to_final.csv").exists():
        return None
    from research.v687 import corpora
    return _count(lambda: corpora.load_buchanan().items, 3722, "concepts")


def _verbnet_make() -> None:
    corpora_dir = _nltk_corpora()
    source = corpora_dir / "verbnet3" if corpora_dir else None
    if not source or not source.is_dir():
        raise Failed("nltk's verbnet3 is not installed: "
                     "python -m nltk.downloader verbnet3")
    out = DATA / "verbnet3.3"
    out.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.xml"):
        shutil.copy2(path, out / path.name)


def _verbnet_check() -> str | None:
    found = list((DATA / "verbnet3.3").glob("*.xml"))
    if not found:
        return None
    if len(found) < 300:
        raise Failed(f"verbnet: {len(found)} xml, expected ~325")
    return f"{len(found)} verbnet classes"


def _babi_make() -> None:
    archive = _get("https://dl.fbaipublicfiles.com/parlai/babi/babi.tar.gz",
                   DATA / "babi" / "babi.tar.gz", "babi.tar.gz (19 MB)")
    with tarfile.open(archive) as tar:
        tar.extractall(DATA / "babi", filter="data")
    archive.unlink()


def _babi_check() -> str | None:
    valid = DATA / "babi" / "tasks_1-20_v1-2" / "en-valid"
    if not valid.is_dir():
        return None
    found = list(valid.glob("*.txt"))
    if len(found) != 60:
        raise Failed(f"babi en-valid: {len(found)} files, expected 60")
    return f"{len(found)} task files"


def _genericskb_make() -> None:
    from ingestion import genericskb
    genericskb.fetch()


def _genericskb_check() -> str | None:
    found = list((DATA / "genericskb").glob("*.parquet"))
    return f"{len(found)} parquet" if found else None


#: `data/` also holds directories nothing reads any more: stepgame, tomi,
#: propara-leaderboard, ubuntu, ubuntu-ranking-dataset-creator, UD_GUM,
#: WordNet, propbank-frames-3.1. Each was searched for across `research/`
#: and `ingestion/` and has no reference at all (`dbpedia` appears only as
#: relation *names* in `normalize.py`, never as a file). They are named here
#: rather than quietly omitted: a step list that silently lacks something is
#: the failure this whole file exists to prevent.
ABANDONED = ("stepgame", "tomi", "propara-leaderboard", "ubuntu",
             "ubuntu-ranking-dataset-creator", "UD_GUM", "WordNet",
             "propbank-frames-3.1", "dbpedia")


def _wiktionary_make() -> None:
    """English noun senses, flattened out of kaikki's raw extract.

    `learn_wiktionary.usable` wants one flat row per sense -- `word`,
    `gloss`, `tags` -- while upstream is one object per (word, part of
    speech) with glosses nested under `senses`. The script that first did
    this was never committed, which is why the file could not be rebuilt.
    """
    import gzip

    url = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
    out = DATA / "wiktionary" / "english-nouns.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"    streaming {url} (2.7 GB gzipped)", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    kept = read = 0
    started = time.time()
    with urllib.request.urlopen(request, timeout=1800) as response, \
            out.open("w", encoding="utf-8") as sink:
        for line in gzip.GzipFile(fileobj=response):
            read += 1
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("lang_code") != "en" or row.get("pos") != "noun":
                continue
            word = row.get("word") or ""
            for sense in row.get("senses") or ():
                glosses = sense.get("glosses") or ()
                if not glosses:
                    continue
                sink.write(json.dumps({
                    "word": word,
                    "gloss": glosses[0],
                    # absent on many senses; `usable` reads it as a set
                    "tags": sense.get("tags") or []}) + "\n")
                kept += 1
            if read % 500_000 == 0:
                print(f"    {read:,} read, {kept:,} senses kept, "
                      f"{time.time() - started:.0f}s", flush=True)
    print(f"    {read:,} rows read, {kept:,} noun senses kept", flush=True)


def _wiktionary_check() -> str | None:
    out = DATA / "wiktionary" / "english-nouns.jsonl"
    if not out.exists():
        return None
    rows = _lines(out)
    # English Wiktionary has hundreds of thousands of noun senses; a far
    # smaller file means the stream was cut off partway.
    if rows < 100_000:
        raise Failed(f"{rows} senses: the kaikki stream was truncated. "
                     f"Delete the file and regenerate.")
    return f"{rows} noun senses"


def _wikipedia_check() -> str | None:
    out = DATA / "wikipedia" / "intros.jsonl"
    if not out.exists():
        return None
    return f"{_lines(out)} article intros"


def _oewn_make() -> None:
    _get("https://github.com/globalwordnet/english-wordnet/releases/download/"
         "2025-edition/english-wordnet-2025-json.zip",
         DATA / "oewn" / "english-wordnet-2025-json.zip",
         "english-wordnet-2025-json.zip (9.5 MB)")


def _questions_make() -> None:
    """The `--external` question datasets, as far as they can be sourced.

    QA-SRL 2.1 is a plain archive. WikiAnswers is the first 20,000 clusters
    of a 30-million-cluster corpus, so it is truncated here rather than
    downloaded whole (40 GB decompressed). Quora's pairs were read here too
    until 2026-09-16, when they were dropped for having no traceable
    source; `teach_reader._natural` no longer reads them.
    """
    import gzip

    out = DATA / "questions"
    out.mkdir(parents=True, exist_ok=True)

    archive = out / "qasrl-v2_1.tar"
    if not archive.exists():
        _get("https://qasrl.org/data/qasrl-v2_1.tar", archive,
             "qasrl-v2_1.tar (37.7 MB)")
    with tarfile.open(archive) as tar:
        tar.extractall(out / "qasrl", filter="data")

    wiki = out / "wikianswers-20k.jsonl"
    if not wiki.exists():
        url = ("https://huggingface.co/datasets/embedding-data/WikiAnswers/"
               "resolve/main/WikiAnswers.jsonl.gz")
        print("    streaming WikiAnswers, keeping the first 20,000 clusters",
              flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(request, timeout=1800) as response, \
                wiki.open("w", encoding="utf-8") as sink:
            for at, line in enumerate(gzip.GzipFile(fileobj=response)):
                if at >= 20_000:
                    break
                sink.write(line.decode("utf-8"))


def _questions_check() -> str | None:
    out = DATA / "questions"
    wiki = out / "wikianswers-20k.jsonl"
    qasrl = out / "qasrl" / "qasrl-v2_1" / "expanded" / "train.jsonl.gz"
    if not wiki.exists() and not qasrl.exists():
        return None
    missing = [name for name, path in (("wikianswers", wiki),
                                       ("qasrl", qasrl)) if not path.exists()]
    if missing:
        raise Failed(f"missing {', '.join(missing)} -- "
                     f"see data/questions.SOURCE.md")
    rows = _lines(wiki)
    if rows != 20_000:
        raise Failed(f"wikianswers-20k.jsonl has {rows} clusters, not 20,000")
    return f"{rows} clusters, qasrl 2.1"


def _memory_check(path: Path, glosses: int, defined: int
                  ) -> Callable[[], str | None]:
    """A definitions memory, against what it held on 2026-09-16.

    Floors, not equalities: these are written by reading glosses with
    v689's own reader, and a rebuild reads with `reader-first` rather than
    the reader that wrote them, so some rows move. Far fewer means the run
    stopped partway -- and a run that stopped partway still leaves a
    perfectly openable database, which is why existence proves nothing.
    """
    def check() -> str | None:
        if not path.exists():
            return None
        import sqlite3

        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            counts = {name: connection.execute(
                f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("glosses", "defined")}
        finally:
            connection.close()
        if counts["glosses"] < glosses * 0.8:
            raise Failed(f"{path.name}: {counts['glosses']} glosses against "
                         f"{glosses} on 2026-09-16 -- a partial run")
        return f"{counts['glosses']} glosses, {counts['defined']} defined"
    return check


def _oewn_check() -> str | None:
    # `ingestion/oewn.py:34` opens this exact name and reads it in place.
    archive = DATA / "oewn" / "english-wordnet-2025-json.zip"
    if not archive.exists():
        return None
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
    except zipfile.BadZipFile as bad:
        raise Failed(f"oewn archive is not a zip: {bad}") from bad
    return f"{len(names)} files, {archive.stat().st_size // 1024 ** 2} MB"


CONCEPTNET = DATA / "conceptnet-assertions-5.7.0.csv.gz"


def _conceptnet_make() -> None:
    _get("https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
         "conceptnet-assertions-5.7.0.csv.gz", CONCEPTNET,
         "conceptnet-assertions-5.7.0.csv.gz (475 MB)")


def _conceptnet_check() -> str | None:
    if not CONCEPTNET.exists():
        return None
    size = CONCEPTNET.stat().st_size // 1024 ** 2
    if size < 400:
        raise Failed(f"conceptnet assertions are only {size} MB")
    return f"{size} MB"


#: PIQA's splits and how many items each documents.
PIQA = {"train": 16113, "valid": 1838}


def _piqa_make() -> None:
    for split in PIQA:
        for name in (f"{split}.jsonl", f"{split}-labels.lst"):
            _get(f"https://yonatanbisk.com/piqa/data/{name}",
                 DATA / "piqa" / name, name)


def _piqa_check() -> str | None:
    if not (DATA / "piqa" / "valid-labels.lst").exists():
        return None
    for split, expected in PIQA.items():
        for name in (f"{split}.jsonl", f"{split}-labels.lst"):
            got = _lines(DATA / "piqa" / name)
            if got != expected:
                raise Failed(f"piqa {name}: {got} lines, documented "
                             f"{expected}")
    return f"{PIQA['train']} train, {PIQA['valid']} dev"


#: MultiPL-E's TypeScript configs and how many tasks each documents.
MULTIPL_E = {"humaneval-ts": 159, "mbpp-ts": 390}


def _humanevalfix_check() -> str | None:
    path = DATA / "humanevalfix" / "js.jsonl"
    if not path.exists():
        return None
    got = _lines(path)
    if got != 164:
        raise Failed(f"humanevalfix: {got} bugs, documented 164")
    return "164 bugs"


def _multipl_e_check() -> str | None:
    folder = DATA / "multipl-e"
    if not (folder / "mbpp-ts.jsonl").exists():
        return None
    for config, expected in MULTIPL_E.items():
        got = _lines(folder / f"{config}.jsonl")
        if got != expected:
            raise Failed(f"multipl-e {config}: {got}, documented {expected}")
    return ", ".join(f"{count} {name}" for name, count in MULTIPL_E.items())


def _actions_check() -> str | None:
    import sqlite3
    path = DATA / "v695_actions.sqlite"
    if not path.exists():
        return None
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        count = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        connection.close()
    if count != 126454:
        raise Failed(f"v695 actions: {count} edges, documented 126454")
    return f"{count} edges"


def _store_check() -> str | None:
    from research.v687 import build
    if not build.DEFAULT_STORE.exists():
        return None
    size = build.DEFAULT_STORE.stat().st_size // 1024 ** 2
    if size < 100:
        raise Failed(f"store is only {size} MB")
    return f"{size} MB"


def _store_make() -> None:
    from research.v687 import build
    build.ensure()


def _minilm_make() -> None:
    from transformers import AutoModel, AutoTokenizer
    out = LLM / "MiniLM-L6-v2"
    out.mkdir(parents=True, exist_ok=True)
    name = "sentence-transformers/all-MiniLM-L6-v2"
    AutoTokenizer.from_pretrained(name).save_pretrained(str(out))
    AutoModel.from_pretrained(name).save_pretrained(str(out))


def _hf_make(repo: str, out: Path) -> Callable[[], None]:
    def make() -> None:
        from huggingface_hub import snapshot_download
        out.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=str(out),
                          ignore_patterns=["*.onnx", "onnx/*", "*.gguf",
                                           "*.msgpack", "*.h5"])
    return make


def _model_check(out: Path, least_mb: int) -> Callable[[], str | None]:
    def check() -> str | None:
        if not (out / "config.json").exists():
            return None
        size = sum(path.stat().st_size for path in out.rglob("*")
                   if path.is_file()) // 1024 ** 2
        if size < least_mb:
            raise Failed(f"{out.name}: {size} MB, expected at least {least_mb}")
        return f"{size} MB"
    return check


def _run(module: str, *args: str, reader: Path | None = None) -> None:
    """`python -m module args`, optionally reading with a given checkpoint.

    Anything that plays a conversation needs a reader, and resolves it
    through `encoder.MODEL`, which is `llm/reader` unless
    `V689_READER_MODEL` says otherwise. During a rebuild from empty there is
    no `llm/reader` yet -- that is the last step of the cycle -- so the
    steps before it must be told to read with `reader-first`.
    """
    import os

    command = [sys.executable, "-u", "-m", module, *args]
    print(f"    {' '.join(command[3:])}", flush=True)
    environment = dict(os.environ)
    if reader is not None:
        environment["V689_READER_MODEL"] = str(reader)
        print(f"    reading with {reader.name}", flush=True)
    result = subprocess.run(command, cwd=str(ROOT), env=environment)
    if result.returncode:
        raise Failed(f"{module} {' '.join(args)} exited {result.returncode}")


def _reader_corpus_make() -> None:
    _run("research.v689.teach_reader", "corpus")


def _reader_corpus_check() -> str | None:
    train = LLM / "reader-data" / "train.jsonl"
    if not train.exists():
        return None
    rows = _lines(train)
    # A `--limit` run writes here too, and training on it would teach the
    # shipped reader from a handful of sources.
    if rows < 20_000:
        raise Failed(f"reader corpus has {rows} rows: that is a --limit run, "
                     f"not the whole corpus. Delete llm/reader-data and "
                     f"regenerate.")
    return f"{rows} rows"


def _reader_train(out: Path) -> Callable[[], None]:
    def make() -> None:
        _run("research.v689.teach_reader", "train",
             "--base", str(LLM / "MiniLM-L6-v2"), "--out", str(out),
             "--epochs", "10")
    return make


def _reader_check(out: Path) -> Callable[[], str | None]:
    def check() -> str | None:
        if not (out / "config.json").exists():
            return None
        # The real test is that it reads: a checkpoint that loads but parses
        # nothing is worse than none, because nothing else will notice.
        # In a separate process, so that pointing the encoder at this
        # checkpoint does not leak into every later step of the run -- the
        # encoder caches the model it loaded, keyed by the path it saw.
        script = ("import os, sys;"
                  f"os.environ['V689_READER_MODEL']={str(out)!r};"
                  "from research.v687 import build;"
                  "from research.v687.language import Parser;"
                  "from research.v687.reason import Reasoner;"
                  "r=Reasoner(build.DEFAULT_STORE);"
                  "p=Parser(vocabulary=r.vocabulary(),"
                  " nouns=r.noun_vocabulary());"
                  "q=p.parse('is a whale a fish');"
                  "r.connection.close();"
                  "print(q.subject, q.relation, q.target)")
        result = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                                capture_output=True, text=True)
        read = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if read[0] != "whale is_a fish":
            raise Failed(f"{out.name} misreads `is a whale a fish` as "
                         f"{read[0]!r}: {(result.stderr or '')[-200:]}")
        return "reads `is a whale a fish` correctly"
    return check


def _turns_check() -> str | None:
    from research.v690.conversations import TURNS
    if not TURNS.exists():
        return None
    turns = _lines(TURNS)
    # DESIGN.md:549 -- 1,851 conversations, 16,427 turns, 14,042 distinct
    # messages. That run was played with the reader that shipped then; this
    # one is played with `reader-first`, so some answers differ and the
    # count with them. Hence a floor and the documented figure beside it,
    # rather than equality (which a healthy run would fail) or mere
    # existence (which a truncated one would pass -- `conversations.run`
    # appends to this file and skips what it already has).
    if turns < 12_000:
        raise Failed(f"{turns} turns against a documented 16,427: a partial "
                     f"run. It appends, so delete llm/decoder-data/"
                     f"turns.jsonl before regenerating.")
    return f"{turns} turns (documented 16,427)"


def _replies_check() -> str | None:
    from research.v690.teach_decoder import REPLIES
    if not REPLIES.exists():
        return None
    rows = _lines(REPLIES)
    # The teacher writes to a *stratified sample*, not to everything:
    # `messages_from` applies `teach_decoder.CAPS` (noted and unknown 1,500,
    # value 1,800, the rest uncapped), which is 6,728 of the 14,151. The
    # other 7,423 are `bootstrap`'s, replied to by the decoder itself.
    # A floor of 10,000 here would have rejected a correct run forever.
    if rows < 6_000:
        raise Failed(f"{rows} replies against a target of about 6,728 "
                     f"(`teach_decoder.CAPS`): unfinished or stopped. It "
                     f"appends, so let it finish, or delete "
                     f"llm/decoder-data/replies.jsonl and start again.")
    return f"{rows} replies (target ~6,728)"


def _label_check() -> str | None:
    found = list((LLM / "reader-data").glob("*-reply.jsonl"))
    if not found:
        return None
    rows = sum(_lines(path) for path in found)
    # This is the *first* label, over the teacher's replies alone: about
    # 6,700 of which some 5,300 trace. DESIGN.md:563's 12,389 is the figure
    # *after* `bootstrap`, and belongs to `label-all`, not here -- a floor
    # of 8,000 against it failed a perfectly good run at 6,268.
    if rows < 4_000:
        raise Failed(f"{rows} labelled replies against about 6,300 from the "
                     f"teacher's alone: a partial run, or `replies` was "
                     f"incomplete when this read it")
    return f"{rows} labelled, the teacher's replies"


def _screened_check() -> str | None:
    path = DATA / "xcslb" / "comps_screened.jsonl"
    if not path.exists():
        return None
    rows = _lines(path)
    # `data/xcslb.SOURCE.md` -- 20,925 of the 49,340 pairs kept, because a
    # calibrated judge denied the foil.
    if rows < 15_000:
        raise Failed(f"{rows} screened pairs against a documented 20,925: "
                     f"a partial run")
    return f"{rows} screened pairs (documented 20,925)"


def _bootstrap_check() -> str | None:
    from research.v690.teach_decoder import BOOTSTRAPPED
    if not BOOTSTRAPPED.exists():
        return None
    rows = _lines(BOOTSTRAPPED)
    # `bootstrap` sees all 14,151 messages (`caps={}`) but skips every one
    # the teacher already replied to (`_kept(path) | _kept(REPLIES)`), so it
    # writes only the rest: 14,151 - 6,728 = 7,423, matching DESIGN.md:553's
    # "the 7,625 messages the teacher never saw". A floor of 8,000 against
    # 14,151 counted the teacher's share twice and failed a complete run.
    if rows < 6_000:
        raise Failed(f"{rows} bootstrapped replies against about 7,400 (the "
                     f"messages the teacher did not reply to): unfinished or "
                     f"stopped. It appends, so let it finish.")
    return f"{rows} bootstrapped replies (the teacher's remainder)"


def _social_check() -> str | None:
    found = sorted((LLM / "reader-data").glob("*-social.jsonl"))
    if not found:
        return None
    return f"{sum(_lines(path) for path in found)} social rows"


def _newer(what: Path, than: Path) -> bool:
    return what.stat().st_mtime > than.stat().st_mtime


def _label_all_check() -> str | None:
    """Labelled *after* the bootstrapped replies existed.

    `label` writes the same files whether it read the teacher's replies
    alone or both sets, so their presence cannot tell the two runs apart --
    the first `label` would satisfy a check that only looked. What
    distinguishes them is recency: the labels must be newer than the
    bootstrapped replies they are supposed to include.
    """
    from research.v690.teach_decoder import BOOTSTRAPPED
    found = list((LLM / "reader-data").glob("*-reply.jsonl"))
    if not found or not BOOTSTRAPPED.exists():
        return None
    if not all(_newer(path, BOOTSTRAPPED) for path in found):
        return None
    rows = sum(_lines(path) for path in found)
    # These are reply *rows* -- up to `MOST_REPLIES` kept per message -- not
    # messages. DESIGN.md:563's 12,389 counts messages read back; the
    # rebuild of 2026-09-16 read back 12,444 messages as 20,949 rows. Rows
    # are what is on disk, so the floor is on rows, and the message says so
    # rather than setting one count beside the other.
    if rows < 15_000:
        raise Failed(f"{rows} reply rows over both sets, against about "
                     f"21,000 (12,444 messages read back): `bootstrap` was "
                     f"incomplete when this read it")
    return f"{rows} reply rows, both sets"


def _decoder_final_check() -> str | None:
    """The decoder trained after the full corpus, not the first one.

    Both trainings write `llm/decoder`, so the same recency argument as
    `_label_all_check` applies: weights older than the labels they were
    meant to learn from are the first decoder, not the last.

    Recency against the labels alone is not enough. `decoder-first` writes
    these very weights *before* `bootstrap` runs, so from the moment it
    finishes until `label-all` rewrites the labels, the weights are newer
    than every label file on disk and the first decoder passes for the
    last. A full run in order happens to recover; `--only decoder` and
    `--list` do not. So the bootstrapped replies must exist and predate the
    weights as well: nothing is the final decoder before the corpus it is
    trained on exists.
    """
    from research.v690.teach_decoder import BOOTSTRAPPED
    weights = LLM / "decoder" / "config.json"
    found = list((LLM / "reader-data").glob("*-reply.jsonl"))
    if not weights.exists() or not found or not BOOTSTRAPPED.exists():
        return None
    if not _newer(weights, BOOTSTRAPPED):
        return None
    if not all(_newer(weights, path) for path in found):
        return None
    return _model_check(LLM / "decoder", 600)()


def _math_corpus_check() -> str | None:
    path = LLM / "math-data" / "train-math.jsonl"
    if not path.exists():
        return None
    rows = _lines(path)
    # `research/v692/corpus.py` at its default 100,000: 70% across the
    # curriculum's acts, the rest utterances with no mathematics to do, a
    # tenth held out.
    if rows < 80_000:
        raise Failed(f"{rows} maths records against about 90,000: a --count "
                     f"run. Delete llm/math-data and regenerate.")
    return f"{rows} records"


def _math_speaking_check() -> str | None:
    pairs = LLM / "math-decoder-data" / "train-math.jsonl"
    answers = LLM / "math-data" / "train-answer.jsonl"
    if not pairs.exists() or not answers.exists():
        return None
    rows = _lines(pairs)
    if rows < 6_000:
        raise Failed(f"{rows} maths message/reply pairs against about 7,200 "
                     f"(`speaking.py` at 8,000, a tenth held out)")
    return f"{rows} pairs, {_lines(answers)} answers to read back"


def _design_corpus_check() -> str | None:
    path = LLM / "design-data" / "train-design.jsonl"
    if not path.exists():
        return None
    rows = _lines(path)
    # `research/v693/stating.py` at its default 20,000 goals and 2,000
    # near misses, a tenth held out.
    if rows < 15_000:
        raise Failed(f"{rows} design records against about 19,800: a "
                     f"--count run. Delete llm/design-data and regenerate.")
    return f"{rows} records"


def _reader_math_check() -> str | None:
    folder = LLM / "reader-design4"
    found = _reader_check(folder)()
    if found is None:
        return None
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    acts = labels.get("heads", {}).get("math_act", {}).get("labels", [])
    if "design" not in acts:
        raise Failed(f"{folder} does not read design goals: it was trained "
                     f"without --subject math --subject design")
    return found + ", and reads mathematics and design goals"


def _decoder_math_check() -> str | None:
    folder = LLM / "decoder-maths"
    if not (folder / "config.json").exists():
        return None
    said = json.loads((folder / "decoder.json").read_text(encoding="utf-8"))
    if "math" not in said.get("subjects", ()):
        raise Failed("llm/decoder-maths was taught without --subject math")
    return _model_check(folder, 600)()


def _designer_proposer_check() -> str | None:
    path = LLM / "designer-proposer" / "model.json"
    if not path.exists():
        return None
    model = json.loads(path.read_text(encoding="utf-8"))
    # `research/v693/proposer.py`: one model for each of the six forms.
    if len(model) < 6:
        raise Failed(f"llm/designer-proposer has {len(model)} form models "
                     f"of 6: a partial run")
    return f"{len(model)} form models"


def _open_designer_proposer_check() -> str | None:
    path = LLM / "open-designer-proposer2" / "model.json"
    if not path.exists():
        return None
    model = json.loads(path.read_text(encoding="utf-8"))
    # `research/v694/proposer.py`: one model for each way (`ways.WAYS`).
    if len(model) < 9:
        raise Failed(f"llm/open-designer-proposer2 has {len(model)} way "
                     f"models of 9: a partial run")
    return f"{len(model)} way models"


def _code_meaning_make() -> None:
    """v696's reader of meaning is taught from: the library's JSDoc,
    MBPP-TS solved by SmolLM3 (kept by the tests), HumanEval-TS solved the
    same way to measure with, generated programs said in English (kept by
    a round trip), then the records."""
    for job in (("docs",), ("solutions",), ("solutions", "--held"),
                ("described", "--count", "4000"), ("corpus",)):
        _run("research.v696.teach_meaning", *job)


def _code_meaning_check() -> str | None:
    """SmolLM3 samples, so a rebuild is not the same corpus line for line:
    what is checked is that every source is there in about its size."""
    path = DATA / "code-meaning" / "corpus.jsonl"
    if not path.exists():
        return None
    import json
    counts: dict = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    least = {"docs": 150, "mbpp-ts": 350, "humaneval-ts": 150,
             "generated": 3000}
    for source, count in least.items():
        if counts.get(source, 0) < count:
            raise Failed(f"code-meaning: {counts.get(source, 0)} {source} "
                         f"records, expected at least {count}")
    return ", ".join(f"{count} {name}" for name, count in counts.items())


def _meaning_check(out: Path) -> Callable[[], str | None]:
    def check() -> str | None:
        if not (out / "heads.bin").exists():
            return None
        import json
        labels = json.loads((out / "labels.json").read_text(
            encoding="utf-8"))
        if len(labels.get("uses", ())) < 60:
            raise Failed(f"{out.name}: {len(labels.get('uses', ()))} "
                         f"things a program uses, expected at least 60")
        return ", ".join(f"{len(names)} {head}"
                         for head, names in labels.items())
    return check


def _sketches_check() -> str | None:
    path = DATA / "code-meaning" / "sketches.jsonl"
    if not path.exists():
        return None
    import json
    counts: dict = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    if counts.get("generated", 0) < 3000 or counts.get("mbpp-ts", 0) < 50:
        raise Failed(f"code-meaning sketches: {counts}, expected at least "
                     f"3000 generated and 50 MBPP programs")
    return ", ".join(f"{count} {name}" for name, count in counts.items())


def _projects_check() -> str | None:
    """Rung 5's frozen project tasks: made once from the verified programs
    (`research/v696/projects.py`); both sets, all three kinds."""
    import json
    counts = {}
    for part in ("dev", "held"):
        path = DATA / "code-meaning" / f"projects-{part}.jsonl"
        if not path.exists():
            return None
        kinds = [json.loads(line)["kind"]
                 for line in path.open(encoding="utf-8")]
        if len(set(kinds)) != 3 or len(kinds) < 30:
            raise Failed(f"projects-{part}: {len(kinds)} tasks of kinds "
                         f"{sorted(set(kinds))}")
        counts[part] = len(kinds)
    return ", ".join(f"{count} {part}" for part, count in counts.items())


def _sketches_check_functions() -> str | None:
    path = DATA / "code-meaning" / "sketches-functions.jsonl"
    if not path.exists():
        return None
    import json
    mbpp = sum(1 for line in path.open(encoding="utf-8")
               if json.loads(line)["source"] == "mbpp-ts")
    if mbpp < 100:
        raise Failed(f"sketches-functions: {mbpp} MBPP functions, expected "
                     f"at least 100")
    return f"{mbpp} MBPP functions"


def _sketches_check_functions2() -> str | None:
    path = DATA / "code-meaning" / "sketches-functions2.jsonl"
    if not path.exists():
        return None
    import json
    mbpp = sum(1 for line in path.open(encoding="utf-8")
               if json.loads(line)["source"] == "mbpp-ts")
    if mbpp < 250:
        raise Failed(f"sketches-functions2: {mbpp} MBPP functions, expected "
                     f"at least 250 (298 when made)")
    return f"{mbpp} MBPP functions"


def _requests_check() -> str | None:
    path = DATA / "code-meaning" / "requests.jsonl"
    if not path.exists():
        return None
    import json
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len(rows) < 600:
        return None   # resumable: not finished (610 when first made)
    return f"{len(rows)} requests, {sum(bool(r['code']) for r in rows)} kept"


def _people_check() -> str | None:
    path = DATA / "code-meaning" / "sketches-people.jsonl"
    if not path.exists():
        return None
    count = sum(1 for _ in path.open(encoding="utf-8"))
    if count < 450:
        raise Failed(f"sketches-people: {count} records, expected 450+ "
                     f"(479 when made)")
    return f"{count} records"


def _risk_labels_check() -> str | None:
    path = DATA / "code-meaning" / "risk.jsonl"
    if not path.exists():
        return None
    import json
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    with_u = sum("U" in row["labels"] for row in rows)
    if with_u < 580:
        # resumable: U still being read (584 of 612 when first made -- the
        # run stopped at its time limit; the rest are teacher's requests)
        return None
    if len(rows) < 4700:
        raise Failed(f"risk: {len(rows)} labelled, expected 4700+ "
                     f"(4772 when made)")
    return f"{len(rows)} labelled, {with_u} with U"


def _risk_estimators_check() -> str | None:
    out = LLM / "risk-estimators"
    if not (out / "risk.json").exists():
        return None
    if not (out / "heads.pt").exists():
        raise Failed("risk-estimators: risk.json without heads.pt")
    return "six heads"


def _code_talk_check() -> str | None:
    path = LLM / "code-talk-data" / "train-code.jsonl"
    if not path.exists():
        return None
    count = sum(1 for _ in path.open(encoding="utf-8"))
    if count < 24000:
        raise Failed(f"code-talk: {count} records, expected 24000+ "
                     f"(24678 when made)")
    return f"{count} records"


VSIX = ROOT / "tools" / "vscode-graph-topology"


def _vsix_make() -> None:
    subprocess.run([sys.executable, str(VSIX / "package.py")], check=True)


def _vsix_check() -> str | None:
    found = sorted((VSIX / "dist").glob("*.vsix"))
    if not found:
        return None
    return f"{found[-1].name}, {found[-1].stat().st_size // 1024} KB"


def _lines_check(path: Path, least: int, made: str
                 ) -> Callable[[], str | None]:
    """A data file there, with at least `least` lines (`made`: how many
    when it was made)."""
    def check() -> str | None:
        if not path.exists():
            return None
        count = sum(1 for _ in path.open(encoding="utf-8"))
        if count < least:
            raise Failed(f"{path.name}: {count} lines, expected {least}+ "
                         f"({made})")
        return f"{count} lines"
    return check


def _sketcher_check(out: Path) -> Callable[[], str | None]:
    def check() -> str | None:
        if not (out / "sketcher.json").exists():
            return None
        return _model_check(out, 600)()
    return check


def steps() -> list[Step]:
    """Every artefact, in the order it can be built."""
    return [
        # -- corpora: downloads, minutes at most ----------------------------
        Step("awa2", "AwA2 class/attribute table (32 KB)",
             _awa2_make, _awa2_check, cost="seconds"),
        Step("xcslb", "COMPS/XCSLB property norms (14 MB)",
             _xcslb_make, _xcslb_check, cost="a minute"),
        Step("entailmentbank", "EntailmentBank proof trees (7.8 MB)",
             _entailmentbank_make, _entailmentbank_check, cost="seconds"),
        Step("thingsplus", "THINGSplus ratings and categories (10 MB)",
             _thingsplus_make, _thingsplus_check, cost="a minute"),
        Step("newton", "NEWTON physical attributes + LVIS categories",
             _newton_make, _newton_check, cost="seconds"),
        Step("buchanan", "Buchanan feature production norms (1.4 MB)",
             _buchanan_make, _buchanan_check, cost="seconds"),
        Step("verbnet", "VerbNet 3.3, copied from nltk_data",
             _verbnet_make, _verbnet_check, cost="seconds"),
        Step("babi", "bAbI tasks 1-20 (19 MB), read by the reader corpus",
             _babi_make, _babi_check, cost="seconds"),
        Step("genericskb", "GenericsKB-Best (39 MB)",
             _genericskb_make, _genericskb_check, cost="a minute"),
        Step("conceptnet", "ConceptNet 5.7 assertions (475 MB)",
             _conceptnet_make, _conceptnet_check, cost="a few minutes"),
        Step("piqa", "PIQA train and dev with labels (6 MB), v695's benchmark",
             _piqa_make, _piqa_check, cost="seconds"),
        Step("multipl-e", "MultiPL-E HumanEval-TS and MBPP-TS, v696's",
             lambda: _run("research.v696.tasks", "fetch"),
             _multipl_e_check, cost="a minute"),
        Step("humanevalfix", "HumanEvalPack's 164 human-written bugs "
                             "(JavaScript), v696 rung 4's held benchmark",
             lambda: _run("research.v696.bugs", "fetch"),
             _humanevalfix_check, cost="seconds"),
        Step("humanevalfix-py", "the same bugs in Python (v699 rung 4)",
             lambda: _run("research.v696.bugs", "fetch", "python"),
             _lines_check(DATA / "humanevalfix" / "python.jsonl", 160,
                          "164 when made"),
             cost="seconds"),
        Step("commitpackft", "CommitPackFT's Python and TypeScript commits: "
                             "what people say a change is, and the change "
                             "(v700's editor)",
             lambda: _run("research.v700.teach_editor", "fetch"),
             _lines_check(DATA / "commitpackft" / "python.jsonl", 56000,
                          "56025 when fetched"),
             cost="a minute"),

        # -- what the ingestion memories are read from ----------------------
        # `state/` survived the 2026-09-16 deletion, so these are usually
        # already present; they are here because the memories cannot be
        # rebuilt without them.
        Step("wiktionary", "English noun senses, flattened out of kaikki",
             _wiktionary_make, _wiktionary_check, cost="30 minutes, 2.7 GB"),
        Step("wikipedia", "article intros, from the live MediaWiki API",
             lambda: _run("ingestion.wikipedia"), _wikipedia_check,
             cost="an hour (it sleeps between batches)"),
        Step("oewn", "Open English WordNet 2025, the changed definitions",
             _oewn_make, _oewn_check, cost="a minute"),
        Step("questions", "WikiAnswers and QA-SRL, read only with --external",
             _questions_make, _questions_check, cost="ten minutes"),

        # -- the taught memories in `state/` --------------------------------
        # Read with v689's own reader, so they come after `reader-first`.
        Step("definitions-memory", "WordNet glosses read into facts",
             lambda: _run("research.v689.learn_definitions", "read",
                          reader=LLM / "reader-first"),
             _memory_check(STATE / "v689-definitions.sqlite", 82116, 69419),
             needs=("store", "reader-first"), cost="an hour"),
        Step("wiktionary-memory", "Wiktionary senses matched to one synset",
             lambda: _run("research.v689.learn_wiktionary", "read",
                          reader=LLM / "reader-first"),
             _memory_check(STATE / "v689-wiktionary.sqlite", 21941, 16918),
             needs=("wiktionary", "store", "reader-first"), cost="an hour"),
        Step("articles-memory", "Wikipedia leads read into facts",
             lambda: _run("research.v689.articles", "read",
                          reader=LLM / "reader-first"),
             _memory_check(STATE / "v689-articles.sqlite", 485, 803),
             needs=("wikipedia", "store", "reader-first"), cost="20 minutes"),

        # -- the store ------------------------------------------------------
        Step("store", "the reasoning store, built from v633 + Ascent++",
             _store_make, _store_check, needs=("awa2", "xcslb"),
             cost="a few minutes"),
        Step("actions", "ConceptNet's doings kept as said (v695)",
             lambda: _run("research.v695.mined"), _actions_check,
             needs=("conceptnet",), cost="a minute"),

        # -- base models ----------------------------------------------------
        Step("minilm", "MiniLM-L6-v2, the reader's base (87 MB)",
             _minilm_make, _model_check(LLM / "MiniLM-L6-v2", 80),
             cost="a minute"),
        Step("smollm2", "SmolLM2-360M-Instruct, the decoder's base (694 MB)",
             _hf_make("HuggingFaceTB/SmolLM2-360M-Instruct",
                      LLM / "SmolLM2-360M-Instruct"),
             _model_check(LLM / "SmolLM2-360M-Instruct", 600),
             cost="a few minutes"),
        Step("smollm3", "SmolLM3-3B, the offline teacher (5.9 GB)",
             _hf_make("HuggingFaceTB/SmolLM3-3B", LLM / "SmolLM3-3B"),
             _model_check(LLM / "SmolLM3-3B", 5000),
             cost="ten minutes or more"),

        # -- the cycle ------------------------------------------------------
        Step("reader-corpus", "what the rules read, for the encoder to learn",
             _reader_corpus_make, _reader_corpus_check,
             needs=("store", "babi"), cost="20 minutes"),
        Step("reader-first", "a reader taught only by the rules",
             _reader_train(LLM / "reader-first"),
             _reader_check(LLM / "reader-first"),
             needs=("reader-corpus", "minilm"), cost="an hour", gpu=True),
        Step("turns", "conversations played through a working engine",
             lambda: _run("research.v690.conversations",
                          reader=LLM / "reader-first"),
             _turns_check, needs=("reader-first",), cost="an hour"),
        # The cycle, exactly as `v690/README.md` "How it is taught" runs it:
        #   replies -> label -> decoder-first -> bootstrap -> label-all
        #   -> reader-social -> reader -> decoder
        # The teacher writes to a stratified sample only (`CAPS`), greedily
        # and one reply a message (`--samples 1`, since `writer.write` takes
        # `greedy=samples == 1`); the first decoder then replies to all
        # 14,151 itself, and what traces is taught beside the teacher's.
        # SmolLM3 wants the card to itself: the README measures 3.7 messages
        # a second alone against 0.2 with a second model resident. Measured
        # here and it holds -- 4.0/s with the card free, against 0.27/s
        # while this file's own `--list` was being run to watch progress,
        # which loads the reader onto the GPU. Watch a run by counting its
        # output file; anything that loads a model makes the run it is
        # watching fifteen times slower.
        Step("replies", "SmolLM3 replying to a stratified sample",
             lambda: _run("research.v690.teach_decoder", "replies",
                          "--samples", "1", "--batch", "12",
                          reader=LLM / "reader-first"),
             _replies_check, needs=("turns", "smollm3"),
             cost="~30 minutes (6,728 at 4/s)", gpu=True),
        # Not a gpu step, though it looks like one. `roundtrip.reading_of`
        # calls the encoder, but only behind `roundtrip.enabled()`, which
        # wants every one of `REPLY_HEADS` in the model's `labels.json`.
        # Both label runs read with `reader-first`, which was taught before
        # any reply existed and so has no reply heads: the check is False
        # and labelling falls back to spaCy alone. Measured mid-run at 0 MiB
        # of GPU across twelve ~800 MB processes, in 35 seconds. Marking it
        # `gpu=True` only broke `--skip-gpu`, which would then skip a step
        # that never wanted the card.
        Step("label", "the teacher's replies, kept where they read back",
             lambda: _run("research.v690.teach_decoder", "label",
                          reader=LLM / "reader-first"),
             _label_check, needs=("replies",), cost="a minute"),
        Step("decoder-first", "SmolLM2 taught on the teacher's replies",
             lambda: _run("research.v690.teach_decoder", "train"),
             _model_check(LLM / "decoder", 600),
             needs=("label", "smollm2"), cost="an hour", gpu=True),
        Step("bootstrap", "the first decoder replying to every message",
             lambda: _run("research.v690.teach_decoder", "bootstrap",
                          "--samples", "4", "--batch", "24",
                          reader=LLM / "reader-first"),
             _bootstrap_check, needs=("decoder-first",),
             cost="hours", gpu=True),
        Step("label-all", "both sets of replies labelled together",
             lambda: _run("research.v690.teach_decoder", "label",
                          reader=LLM / "reader-first"),
             _label_all_check, needs=("bootstrap",), cost="20 minutes"),
        Step("reader-social", "the social acts, for the reader to learn",
             lambda: _run("research.v689.teach_reader", "social",
                          reader=LLM / "reader-first"),
             _social_check, needs=("reader-corpus", "store", "reader-first"),
             cost="a minute"),
        Step("reader", "the shipped reader, taught on the replies too",
             _reader_train(LLM / "reader"), _reader_check(LLM / "reader"),
             needs=("label-all", "reader-social", "minilm"),
             cost="an hour", gpu=True),
        Step("decoder", "SmolLM2 taught again on everything that traces",
             lambda: _run("research.v690.teach_decoder", "train"),
             _decoder_final_check, needs=("label-all", "smollm2"),
             cost="an hour", gpu=True),

        # -- mathematics as a subject (research/v692) ----------------------
        # The shipped reader and decoder are taught without it; these are
        # taught with it, into folders of their own, and are what the
        # runtime reads with where they exist (`encoder.MODEL`,
        # `decoder.MODEL`).
        Step("math-corpus", "maths said in words, each word's part and "
                            "symbols",
             lambda: _run("research.v692.corpus"), _math_corpus_check,
             needs=("reader-corpus",), cost="ten minutes"),
        Step("math-speaking", "maths replies for the decoder, and to read "
                              "back",
             lambda: _run("research.v692.speaking"), _math_speaking_check,
             needs=("math-corpus",), cost="five minutes"),
        Step("design-corpus", "design goals said in English, each word's "
                              "part (research/v693)",
             lambda: _run("research.v693.stating"), _design_corpus_check,
             needs=("reader-corpus",), cost="five minutes"),
        Step("reader-math", "the reader taught mathematics and design "
                            "goals too",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "MiniLM-L6-v2"),
                          "--out", str(LLM / "reader-design4"),
                          "--epochs", "10", "--subject", "math",
                          "--subject", "design"),
             _reader_math_check,
             needs=("label-all", "reader-social", "minilm", "math-corpus",
                    "math-speaking", "design-corpus"),
             cost="an hour and a half", gpu=True),
        Step("decoder-math", "the decoder taught to say mathematics too",
             lambda: _run("research.v690.teach_decoder", "train",
                          "--save", str(LLM / "decoder-maths"),
                          "--subject", "math"),
             _decoder_math_check,
             needs=("label-all", "smollm2", "math-speaking"),
             cost="an hour", gpu=True),

        # -- designing (research/v693) -------------------------------------
        Step("designer-proposer", "which design form to try first, taught "
                                  "from goals made from objects",
             lambda: _run("research.v693.proposer", "train"),
             _designer_proposer_check, cost="a few minutes"),

        # -- designing in the open world (research/v694) -------------------
        Step("open-designer-proposer", "which way to a goal to try first, "
                                       "taught from goals made from what "
                                       "the store says things are for",
             lambda: _run("research.v694.proposer", "train"),
             _open_designer_proposer_check, needs=("store", "verbnet"),
             cost="a few minutes"),

        # -- reading code and English together (research/v696) ------------
        Step("unixcoder", "UniXcoder-base, the reader of meaning's base "
                          "(480 MB)",
             _hf_make("microsoft/unixcoder-base", LLM / "unixcoder-base"),
             _model_check(LLM / "unixcoder-base", 400), cost="a minute"),
        Step("code-meaning", "English and code for the reader of meaning, "
                             "every label read exactly",
             _code_meaning_make, _code_meaning_check,
             needs=("multipl-e", "smollm3"), cost="two hours", gpu=True),
        Step("meaning", "the reader of meaning: English and code, one "
                        "sequence, one structure",
             lambda: _run("research.v696.reader", "train", "--base",
                          "unixcoder-base", "--model", "meaning-unixcoder"),
             _meaning_check(LLM / "meaning-unixcoder"),
             needs=("code-meaning", "unixcoder"), cost="ten minutes",
             gpu=True),
        Step("sketches", "programs in the search's own language, for the "
                         "decoder: MBPP solved as one expression by SmolLM3 "
                         "(kept by tests and parsing), and the rest read back",
             lambda: (_run("research.v696.teach_sketch", "expressions"),
                      _run("research.v696.teach_sketch", "corpus")),
             _sketches_check, needs=("meaning", "smollm3"),
             cost="half an hour", gpu=True),
        Step("sketcher", "the decoder: a request and its meaning to programs "
                         "the search checks",
             lambda: _run("research.v696.sketcher", "train", "--epochs", "2",
                          "--model", "sketcher-e2"),
             _sketcher_check(LLM / "sketcher-e2"),
             needs=("sketches", "smollm2"), cost="fifteen minutes",
             gpu=True),
        Step("projects", "rung 5's projects, assembled from verified "
                         "programs: use, bug and change tasks, frozen",
             lambda: _run("research.v696.projects"), _projects_check,
             needs=("code-meaning",), cost="an hour"),
        Step("sketches-functions", "whole functions as written -- steps, "
                                   "loops, helpers -- that read into the "
                                   "search's tree (rung 3)",
             lambda: _run("research.v696.teach_sketch", "corpus",
                          "--functions"),
             _sketches_check_functions, needs=("sketches",),
             cost="five minutes", gpu=True),
        Step("sketcher-functions", "the decoder writing whole functions",
             lambda: _run("research.v696.sketcher", "train", "--functions",
                          "--epochs", "2", "--model", "sketcher-functions"),
             _sketcher_check(LLM / "sketcher-functions"),
             needs=("sketches-functions", "smollm2"),
             cost="twenty-five minutes", gpu=True),
        Step("sketches-functions2", "the same, made again once the reader "
                                    "read more of what people write (nearly "
                                    "twice the MBPP functions)",
             lambda: _run("research.v696.teach_sketch", "corpus",
                          "--functions", "--out",
                          "sketches-functions2.jsonl"),
             _sketches_check_functions2, needs=("sketches",),
             cost="five minutes", gpu=True),
        Step("sketcher-functions2", "the decoder taught from that corpus",
             lambda: _run("research.v696.sketcher", "train", "--functions",
                          "--epochs", "2", "--model", "sketcher-functions2",
                          "--corpus", "sketches-functions2.jsonl"),
             _sketcher_check(LLM / "sketcher-functions2"),
             needs=("sketches-functions2", "smollm2"),
             cost="half an hour", gpu=True),
        Step("requests", "new requests in MultiPL-E's form, written and "
                         "solved by SmolLM3 offline, kept when its solution "
                         "meets their examples and reads",
             lambda: _run("research.v696.teach_requests", "write",
                          "--count", "610"),
             _requests_check, needs=("multipl-e", "smollm3"),
             cost="two hours", gpu=True),
        Step("sketches-people", "the decoder's records as people write "
                                "functions: those requests and MBPP's",
             lambda: _run("research.v696.teach_requests", "corpus"),
             _people_check, needs=("requests", "code-meaning"),
             cost="five minutes"),
        Step("sketcher-people", "a decoder taught those, no meaning line",
             lambda: _run("research.v696.sketcher", "train", "--functions",
                          "--no-meaning", "--epochs", "2", "--model",
                          "sketcher-people", "--corpus",
                          "sketches-people.jsonl"),
             _sketcher_check(LLM / "sketcher-people"),
             needs=("sketches-people", "smollm2"), cost="ten minutes",
             gpu=True),
        Step("risk-labels", "six risks of every request with a verified "
                            "program: read off it, and U from the untaught "
                            "base model's programs for it",
             lambda: _run("research.v696.risk", "label"), _risk_labels_check,
             needs=("code-meaning", "requests", "smollm2"),
             cost="an hour and a half", gpu=True),
        Step("risk-estimators", "six estimators, one per risk, reading the "
                                "request with the reader of meaning",
             lambda: _run("research.v696.risk", "train"),
             _risk_estimators_check, needs=("risk-labels", "meaning"),
             cost="a few minutes", gpu=True),

        # -- the editor (v698) ------------------------------------------------
        Step("code-talk", "messages about code for the reader: written and "
                          "checked by SmolLM3, names of every shape, what is "
                          "not code talk (research/v698)",
             lambda: (_run("research.v698.teach_code_talk", "lists"),
                      _run("research.v698.teach_code_talk", "write"),
                      _run("research.v698.teach_code_talk", "check"),
                      _run("research.v698.teach_code_talk", "check-ways"),
                      _run("research.v698.teach_code_talk", "corpus")),
             _code_talk_check, needs=("multipl-e", "smollm3", "reader-corpus",
                                      "math-corpus", "editor-corpus"),
             cost="three hours", gpu=True),
        Step("reader-code", "the shared reader taught code talk too",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "reader-design4"),
                          "--out", str(LLM / "reader-code7"),
                          "--epochs", "4", "--subject", "math",
                          "--subject", "design", "--subject", "code"),
             _model_check(LLM / "reader-code7", 80),
             needs=("code-talk", "reader-math"), cost="half an hour",
             gpu=True),
        # taught again from reader-code7, not from reader-design4: four
        # epochs from design4 on the corpus with changes of how code is
        # written (`use a switch`, `make it iterative`) lost a v689
        # compound statement; two more from code7 keep the whole suite
        Step("reader-code-ways", "the shared reader taught changes of how "
                                 "code is written (research/v698)",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "reader-code7"),
                          "--out", str(LLM / "reader-code9"),
                          "--epochs", "2", "--subject", "math",
                          "--subject", "design", "--subject", "code"),
             _model_check(LLM / "reader-code9", 80),
             needs=("reader-code",), cost="fifteen minutes", gpu=True),
        # which ways of writing a message asks for (research/v698/ways.py):
        # a head over the reader of meaning, as the risk estimators are --
        # the shared reader is not touched (taught it, v689 lost compound
        # statements each time)
        Step("ways-estimator", "which ways of writing code a message asks "
                               "for, read over the reader of meaning",
             lambda: _run("research.v698.asked_ways", "tune"),
             lambda: (f"chosen {json.loads((LLM / 'ways-estimator' / 'ways.json').read_text(encoding='utf-8'))['chosen']}"
                      if (LLM / "ways-estimator" / "ways.json").exists()
                      else None),
             needs=("code-talk", "meaning"), cost="ten minutes",
             gpu=True),
        Step("vscode-extension", "the VS Code harness, packed as a .vsix "
                                 "(install: package.py --install)",
             _vsix_make, _vsix_check, cost="seconds"),

        # -- v699: Python, fully -------------------------------------------
        Step("multipl-e-py", "the same tasks in Python: MultiPL-E's typed "
                             "originals, MBPP's and HumanEval's solutions",
             lambda: _run("research.v696.tasks", "fetch-python"),
             _lines_check(DATA / "multipl-e" / "mbpp-py.jsonl", 380,
                          "390 when made"),
             needs=("multipl-e",), cost="a few minutes"),
        Step("code-meaning-py", "Python's corpora for the reader of meaning: "
                                "its library's docstrings, people's solutions "
                                "kept by the tests, generated programs said "
                                "in English (SmolLM3, kept by a round trip)",
             lambda: (_run("research.v696.pycorpus", "docs"),
                      _run("research.v696.pycorpus", "solutions"),
                      _run("research.v696.pycorpus", "described",
                           "--count", "4000")),
             # 3856 when made: the run stopped at its time limit, each kept as
             # it was said (a run again goes on from there)
             _lines_check(DATA / "code-meaning" / "described-py.jsonl", 3800,
                          "3856 when made"),
             needs=("multipl-e-py", "smollm3"), cost="two hours", gpu=True),
        Step("requests-py", "new Python requests, written and solved by "
                            "SmolLM3 offline, kept as TypeScript's are",
             lambda: _run("research.v696.teach_requests", "write-python",
                          "--count", "600"),
             _lines_check(DATA / "code-meaning" / "requests-py.jsonl", 550,
                          "600 when made"),
             needs=("multipl-e-py", "smollm3"), cost="two hours", gpu=True),
        Step("meaning-bilingual", "the reader of meaning taught both "
                                  "languages, one vocabulary of what code "
                                  "uses",
             lambda: (_run("research.v696.teach_meaning", "corpus"),
                      _run("research.v696.reader", "train", "--base",
                           "unixcoder-base", "--model",
                           "meaning-bilingual")),
             _meaning_check(LLM / "meaning-bilingual"),
             needs=("code-meaning", "code-meaning-py", "unixcoder"),
             cost="twenty minutes", gpu=True),
        Step("sketcher-functions3", "the writer taught both languages, the "
                                    "language in its system line",
             lambda: (_run("research.v696.teach_sketch", "corpus",
                           "--functions", "--out",
                           "sketches-functions3.jsonl", "--reader",
                           "meaning-bilingual"),
                      _run("research.v696.sketcher", "train", "--functions",
                           "--epochs", "2", "--model", "sketcher-functions3",
                           "--corpus", "sketches-functions3.jsonl")),
             _sketcher_check(LLM / "sketcher-functions3"),
             needs=("meaning-bilingual", "smollm2"), cost="forty minutes",
             gpu=True),
        Step("sketcher-people2", "the people's writer, both languages",
             lambda: (_run("research.v696.teach_requests", "corpus"),
                      _run("research.v696.sketcher", "train", "--functions",
                           "--no-meaning", "--epochs", "2", "--model",
                           "sketcher-people2", "--corpus",
                           "sketches-people.jsonl")),
             _sketcher_check(LLM / "sketcher-people2"),
             needs=("requests", "requests-py", "smollm2"),
             cost="fifteen minutes", gpu=True),
        Step("risk-estimators2", "the six risk estimators over the bilingual "
                                 "reader, Python's requests labelled too",
             lambda: (_run("research.v696.risk", "label"),
                      _run("research.v696.risk", "train", "--model",
                           "risk-estimators2", "--reader",
                           "meaning-bilingual")),
             lambda: ("trained" if (LLM / "risk-estimators2"
                                    / "risk.json").exists() else None),
             needs=("risk-labels", "meaning-bilingual", "requests-py"),
             cost="two hours", gpu=True),
        # the ways estimator taught on the code talk as checked outright
        # (`teach_code_talk._family_choice`: asked which a message is
        # written with, the teacher took any message not naming a regex for
        # `without a regular expression`); -2..-8 were v699's steps there
        # ... and on requests asked without the way their seed names (a task
        # is not a way: `sum the even numbers` was read as TypeScript), in
        # two rounds -- the seeds, then the plain ones said again
        Step("ways-estimator9", "the ways of writing a message asks for, "
                                "in both languages, and which language",
             lambda: (_run("research.v698.teach_code_talk",
                           "contrast-write"),
                      _run("research.v698.teach_code_talk",
                           "contrast-check"),
                      _run("research.v698.teach_code_talk",
                           "contrast-write"),
                      _run("research.v698.teach_code_talk",
                           "contrast-check"),
                      _run("research.v698.asked_ways", "tune", "--out",
                           "ways-estimator9", "--reader",
                           "meaning-bilingual")),
             lambda: (f"chosen {json.loads((LLM / 'ways-estimator9' / 'ways.json').read_text(encoding='utf-8'))['chosen']}"
                      if (LLM / "ways-estimator9" / "ways.json").exists()
                      else None),
             needs=("code-talk", "meaning-bilingual"), cost="half an hour",
             gpu=True),
        Step("ways-held", "the rare ways' own sets, never taught: messages "
                          "asking for each and its near misses, kept as "
                          "the teacher reads them; the estimator measured "
                          "on them",
             lambda: (_run("research.v698.teach_code_talk", "held-write"),
                      _run("research.v698.teach_code_talk", "held-check"),
                      _run("research.v698.asked_ways", "challenge", "--out",
                           "ways-estimator9")),
             _lines_check(LLM / "code-talk-data" / "ways-held.jsonl", 900,
                          "about 1000 when made"),
             needs=("code-talk", "ways-estimator9", "smollm3"),
             cost="twenty minutes", gpu=True),
        Step("reader-claims", "claims said against each other (`so cats "
                              "don't swim, but this cat swam?`), read and "
                              "placed by the rules: a comma before them is "
                              "no time phrase",
             lambda: _run("research.v689.teach_reader", "claims"),
             lambda: (f"{_lines(LLM / 'reader-data' / 'train-claims.jsonl')}"
                      f" claim rows" if (LLM / "reader-data"
                                         / "train-claims.jsonl").exists()
                      else None),
             needs=("store", "reader-corpus"), cost="two minutes"),
        # reader-code10 (without the claims) rephrased v689's pig claim as a
        # question is, `fly` moved to the end; -10..-15 were v698's
        # experiments' names, -16..-19 and -21 v699's tries (DESIGN.md)
        Step("reader-code-python", "the shared reader taught Python's code "
                                   "talk (snake_case names, .py files)",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "reader-code9"),
                          "--out", str(LLM / "reader-code20"),
                          "--epochs", "2", "--subject", "math",
                          "--subject", "design", "--subject", "code"),
             _model_check(LLM / "reader-code20", 80),
             needs=("reader-code-ways", "code-talk", "reader-claims"),
             cost="twenty minutes", gpu=True),
        # v700: changes of the project's code, said as people say them --
        # commit messages (`editor-corpus`) in the code-talk corpus
        Step("editor-corpus", "commits as changes: the message, the part of "
                              "the file changed, the change as blocks",
             lambda: _run("research.v700.teach_editor", "corpus"),
             _lines_check(LLM / "editor-data" / "edits.jsonl", 38000,
                          "41233 when made"),
             needs=("commitpackft",), cost="a minute"),
        Step("editor", "a writer taught to change code as it is asked "
                       "(research/v700/teach_editor.py)",
             lambda: _run("research.v700.teach_editor", "train",
                          "--out", "editor"),
             _model_check(LLM / "editor", 600),
             needs=("editor-corpus",), cost="three hours", gpu=True),
        Step("editor-faults", "the editor taught again: faults found in "
                              "real code, fixed by construction, and "
                              "commits of small changes in a function",
             lambda: (_run("research.v700.teach_faults", "phrase"),
                      _run("research.v700.teach_faults", "rows"),
                      _run("research.v700.teach_editor", "mix"),
                      _run("research.v700.teach_editor", "train",
                           "--base", "editor", "--out", "editor3",
                           "--epochs", "1", "--rate", "5e-5",
                           "--seed", "702", "--corpus", "edits-mix.jsonl")),
             _model_check(LLM / "editor3", 600),
             needs=("editor", "smollm3"), cost="two hours", gpu=True),
        Step("change-judge", "a judge of changes: does it do what was "
                             "said (research/v700/teach_judge.py)",
             lambda: (_run("research.v700.teach_editor", "train",
                           "--base", "editor", "--out", "editor2",
                           "--epochs", "1", "--rate", "5e-5",
                           "--seed", "701"),
                      _run("research.v700.teach_judge", "samples",
                           "--editor", "editor2"),
                      _run("research.v700.teach_judge", "train",
                           "--out", "change-judge2")),
             _model_check(LLM / "change-judge2", 400),
             needs=("editor",), cost="nine hours", gpu=True),
        # v701: questions about data, read by the shared reader
        Step("data-talk", "questions about data files, from real files' "
                          "schemas, said again by SmolLM3 (research/v701)",
             lambda: (_run("research.v701.teach_data_talk", "seeds"),
                      _run("research.v701.teach_data_talk", "write"),
                      _run("research.v701.teach_data_talk", "corpus")),
             _lines_check(LLM / "data-talk-data" / "train-data.jsonl", 5000,
                          "as made"),
             needs=("commitpackft", "smollm3", "code-talk"),
             cost="two hours", gpu=True),
        Step("reader-code-edits", "the shared reader taught changes of the "
                                  "project's code, as commits say them",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "reader-code20"),
                          "--out", str(LLM / "reader-code23"),
                          "--epochs", "2", "--subject", "math",
                          "--subject", "design", "--subject", "code"),
             _model_check(LLM / "reader-code23", 80),
             needs=("reader-code-python", "code-talk", "editor-corpus"),
             cost="twenty minutes", gpu=True),
        Step("reader-data-talk", "the shared reader taught questions about data "
                            "(research/v701)",
             lambda: _run("research.v689.teach_reader", "train",
                          "--base", str(LLM / "reader-code23"),
                          "--out", str(LLM / "reader-code24"),
                          "--epochs", "2", "--subject", "math",
                          "--subject", "design", "--subject", "code",
                          "--subject", "data"),
             _model_check(LLM / "reader-code24", 80),
             needs=("reader-code-edits", "data-talk"),
             cost="twenty-five minutes", gpu=True),

        # -- measurement ----------------------------------------------------
        Step("screened", "COMPS foils a calibrated judge denied",
             lambda: _run("research.v688.screen", "--build"), _screened_check,
             needs=("xcslb", "smollm3"), cost="hours", gpu=True),
    ]


def _ordered(known: dict, wanted: set) -> list[str]:
    """`wanted`, in an order where every step follows what it needs.

    The declared `needs` decide this, not where a step sits in the list.
    Hand-ordering put the taught memories beside the corpora they are read
    from, which reads well and is wrong: they need the store and a reader,
    both declared far below them, so a run from empty attempted them first.
    """
    out: list[str] = []
    done: set[str] = set()
    open_: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in done:
            return
        if name in open_:
            raise Failed("steps need each other in a circle: "
                         + " -> ".join(path + (name,)))
        open_.add(name)
        for need in known[name].needs:
            if need not in known:
                raise Failed(f"{name} needs {need!r}, which is not a step")
            visit(need, path + (name,))
        open_.discard(name)
        done.add(name)
        if name in wanted:
            out.append(name)

    for name in known:
        visit(name, ())
    return out


def main(argv=None) -> int:
    known = {step.name: step for step in steps()}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="what there is, and what is already present")
    parser.add_argument("--only", nargs="+", metavar="STEP",
                        help="these steps and what they need")
    parser.add_argument("--force", nargs="+", metavar="STEP", default=[],
                        help="rebuild these even if they look present")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="stop before anything that trains")
    options = parser.parse_args(argv)

    for name in (options.only or []) + options.force:
        if name not in known:
            parser.error(f"no step {name!r}; try --list")

    wanted = set(known)
    if options.only:
        wanted, queue = set(), list(options.only)
        while queue:
            name = queue.pop()
            if name in wanted:
                continue
            wanted.add(name)
            queue.extend(known[name].needs)
    wanted = _ordered(known, wanted)

    if options.list:
        for name in known:
            step = known[name]
            try:
                state = step.check()
            except Failed as bad:
                state = f"BROKEN: {bad}"
            mark = "ok  " if state and not str(state).startswith("BROKEN") \
                else ("BAD " if state else "-   ")
            print(f"  {mark}{name:14} {state or 'missing':<44} "
                  f"{step.cost}{' [gpu]' if step.gpu else ''}")
            print(f"       {step.what}")
        return 0

    started, built, failures = time.time(), [], []
    for name in wanted:
        step = known[name]
        if options.skip_gpu and step.gpu:
            print(f"[skip] {name}: needs the gpu")
            continue
        try:
            state = None if name in options.force else step.check()
        except Failed as bad:
            state = None
            print(f"[bad ] {name}: {bad}")
        if state:
            print(f"[have] {name}: {state}")
            continue
        print(f"[make] {name}: {step.what} ({step.cost})", flush=True)
        if options.dry_run:
            continue
        try:
            step.make()
            confirmed = step.check()
            if not confirmed:
                raise Failed("made, but its check still says it is missing")
            print(f"[done] {name}: {confirmed}", flush=True)
            built.append(name)
        except Exception as bad:                          # noqa: BLE001
            print(f"[FAIL] {name}: {type(bad).__name__}: {bad}", flush=True)
            failures.append(name)
            break

    print(f"\n{len(built)} built, {len(failures)} failed, "
          f"{round(time.time() - started)}s")
    if failures:
        print(f"failed: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
