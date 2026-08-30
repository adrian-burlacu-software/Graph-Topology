
from __future__ import annotations

from pathlib import Path


def _parse_cols(line):
    cols=line.rstrip("\n").split("\t")
    if len(cols)!=10:
        return None
    return cols


def iter_udgum(root):
    root=Path(root)

    for path in sorted(root.rglob("*.conllu")):
        sent_id=None
        text=None
        tokens=[]

        with path.open("r",encoding="utf-8",errors="replace") as f:
            for raw in f:
                line=raw.rstrip("\n")

                if not line:
                    if tokens:
                        yield {
                            "path":path,
                            "sent_id":sent_id,
                            "text":text or " ".join(x["form"] for x in tokens),
                            "tokens":tokens,
                        }
                    sent_id=None
                    text=None
                    tokens=[]
                    continue

                if line.startswith("# sent_id"):
                    sent_id=line.split("=",1)[-1].strip()
                    continue

                if line.startswith("# text"):
                    text=line.split("=",1)[-1].strip()
                    continue

                if line.startswith("#"):
                    continue

                cols=_parse_cols(line)
                if not cols:
                    continue

                token_id=cols[0]
                # Skip multiword ranges and empty nodes; retain ordinary tokens.
                if "-" in token_id or "." in token_id:
                    continue

                tokens.append({
                    "id":token_id,
                    "form":cols[1],
                    "lemma":cols[2],
                    "upos":cols[3],
                    "xpos":cols[4],
                    "feats":cols[5],
                    "head":cols[6],
                    "deprel":cols[7],
                    "deps":cols[8],
                    "misc":cols[9],
                })
