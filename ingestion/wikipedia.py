"""Wikipedia lead paragraphs for the concepts the audit asks about.

    python -m ingestion.wikipedia --out data/wikipedia/intros.jsonl

Twenty titles a request to the MediaWiki API (`prop=extracts` with
`exintro`, the plain-text introduction), redirects followed, and
disambiguation pages recorded as such rather than read. The titles are the
XCSLB concepts, each tied to the synset XCSLB's own sense key names
(`audit.store_senses`), and the classes the over-affirmation guards ask
about: XCSLB is almost all leaves, and a class article is where `some bats
fly` is written about mammals.

Wikipedia text is CC BY-SA 4.0. It is kept under `data/`, which is not in git.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API = "https://en.wikipedia.org/w/api.php"
AGENT = "GraphTopologyResearch/0.1 (knowledge-graph research, low volume)"
OUT = REPOSITORY_ROOT / "data" / "wikipedia" / "intros.jsonl"

#: `exintro` extracts come twenty pages a request at most.
BATCH = 20

#: (synset, article) for the classes and examples the guards ask about.
CLASSES = (("animal.n.01", "Animal"), ("fish.n.01", "Fish"),
           ("mammal.n.01", "Mammal"), ("bird.n.01", "Bird"),
           ("person.n.01", "Human"), ("rock.n.01", "Rock (geology)"),
           ("dog.n.01", "Dog"), ("cat.n.01", "Cat"), ("horse.n.01", "Horse"),
           ("testis.n.01", "Testicle"), ("kitten.n.01", "Kitten"),
           ("canoe.n.01", "Canoe"), ("mudskipper.n.01", "Mudskipper"),
           ("penguin.n.01", "Penguin"), ("bat.n.01", "Bat"))


def wanted() -> list[tuple[str, str]]:
    """(synset, title), one article per synset."""
    from research.v688.audit import store_senses

    out: dict = {}
    for concept, synset in store_senses().items():
        out.setdefault(synset, concept[:1].upper() + concept[1:])
    for synset, title in CLASSES:
        out[synset] = title
    return sorted(out.items())


def fetch(titles: list[str]) -> dict:
    """title -> {page, missing, disambiguation, extract}."""
    params = {"action": "query", "prop": "extracts|pageprops",
              "exintro": "1", "explaintext": "1", "redirects": "1",
              "format": "json", "exlimit": "max", "titles": "|".join(titles)}
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": AGENT})
    data = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as bad:
            if bad.code not in (429, 503) or attempt == 7:
                raise
            # Asked to slow down: wait as long as told, and longer each time.
            wait = int(bad.headers.get("Retry-After") or 0) or 10 * 2 ** attempt
            print(f"[wikipedia] {bad.code}, waiting {wait}s", flush=True)
            time.sleep(wait)
    query = data.get("query") or {}
    renamed = {step["from"]: step["to"] for step in
               (query.get("normalized") or []) + (query.get("redirects") or [])}
    pages = {page.get("title"): page
             for page in (query.get("pages") or {}).values()}
    out = {}
    for title in titles:
        final = title
        for _ in range(3):
            final = renamed.get(final, final)
        page = pages.get(final) or {}
        out[title] = {"page": page.get("title", ""),
                      "missing": "missing" in page or not page,
                      "disambiguation": "disambiguation" in (
                          page.get("pageprops") or {}),
                      "extract": page.get("extract") or ""}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    options = parser.parse_args(argv)
    items = wanted()
    options.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counts = {"articles": 0, "missing": 0, "disambiguation": 0}
    with options.out.open("w", encoding="utf-8") as handle:
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            found = fetch([title for _, title in batch])
            for synset, title in batch:
                row = {"synset": synset, "title": title, **found[title]}
                counts["missing"] += row["missing"]
                counts["disambiguation"] += row["disambiguation"]
                counts["articles"] += bool(row["extract"]) and not (
                    row["disambiguation"])
                handle.write(json.dumps(row) + "\n")
            print(f"[wikipedia] {min(start + BATCH, len(items))}/{len(items)} "
                  f"{counts} {time.time() - started:.0f}s", flush=True)
            time.sleep(4.0)
    print(json.dumps({"titles": len(items), **counts,
                      "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
