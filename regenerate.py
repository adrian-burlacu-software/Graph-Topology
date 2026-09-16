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
    # SmolLM3 writes a few replies for each of ~14,000 messages and appends
    # as it goes, so a file that exists proves only that the run started.
    # Without this floor a run in progress reports as done -- caught with
    # forty-eight replies in the file, which `label` would have taken and
    # trained a decoder and a reader on.
    if rows < 10_000:
        raise Failed(f"{rows} replies: the run is unfinished or was stopped "
                     f"(~14,000 messages, a few replies each). It appends, "
                     f"so let it finish, or delete "
                     f"llm/decoder-data/replies.jsonl and start again.")
    return f"{rows} replies"


def _label_check() -> str | None:
    found = list((LLM / "reader-data").glob("*-reply.jsonl"))
    if not found:
        return None
    rows = sum(_lines(path) for path in found)
    # DESIGN.md:563 -- 12,389 messages read back, after the narration check
    # dropped 93 of the 12,482 that decoder v3 was taught on.
    if rows < 8_000:
        raise Failed(f"{rows} labelled replies against a documented 12,389: "
                     f"a partial run, or `replies` was incomplete when this "
                     f"read it")
    return f"{rows} labelled replies (documented 12,389)"


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


def steps() -> list[Step]:
    """Every artefact, in the order it can be built."""
    return [
        # -- corpora: downloads, minutes at most ----------------------------
        Step("awa2", "AwA2 class/attribute table (32 KB)",
             _awa2_make, _awa2_check, cost="seconds"),
        Step("xcslb", "COMPS/XCSLB property norms (14 MB)",
             _xcslb_make, _xcslb_check, cost="a minute"),
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
        Step("replies", "SmolLM3 writing a reply to each message",
             lambda: _run("research.v690.teach_decoder", "replies",
                          reader=LLM / "reader-first"),
             _replies_check, needs=("turns", "smollm3"),
             cost="hours", gpu=True),
        Step("label", "replies kept only where they read back",
             lambda: _run("research.v690.teach_decoder", "label",
                          reader=LLM / "reader-first"),
             _label_check, needs=("replies",), cost="20 minutes"),
        Step("decoder", "SmolLM2 fine-tuned to write the replies",
             lambda: _run("research.v690.teach_decoder", "train"),
             _model_check(LLM / "decoder", 600),
             needs=("label", "smollm2"), cost="an hour", gpu=True),
        Step("reader", "the shipped reader, taught on the replies too",
             _reader_train(LLM / "reader"), _reader_check(LLM / "reader"),
             needs=("label", "minilm"), cost="an hour", gpu=True),

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
