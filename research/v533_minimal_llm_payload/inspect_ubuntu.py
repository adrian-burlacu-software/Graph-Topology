
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingest_adapters import (
    load_pickle_robust,
    normalize_ubuntu_vocab,
    ubuntu_container_summary,
    extract_ubuntu_relational,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ubuntu",default=r".\data\ubuntu")
    ap.add_argument("--sample-pairs",type=int,default=5)
    args=ap.parse_args()

    root=Path(args.ubuntu).resolve()
    print(f"[UBUNTU INSPECT] {root}")

    dataset_path=root/"dataset.pkl"
    vocab_path=root/"vocab.pkl"
    w_path=root/"W.pkl"

    dataset=None
    vocab={}

    if dataset_path.exists():
        print(
            f"[UBUNTU] dataset.pkl "
            f"{dataset_path.stat().st_size/1024/1024:.1f}MB"
        )
        dataset=load_pickle_robust(dataset_path)

        print(
            "[UBUNTU] container structure:",
            json.dumps(
                ubuntu_container_summary(dataset),
                indent=2,
            ),
        )

    if vocab_path.exists():
        print(
            f"[UBUNTU] vocab.pkl "
            f"{vocab_path.stat().st_size/1024/1024:.1f}MB"
        )
        obj=load_pickle_robust(vocab_path)
        vocab=normalize_ubuntu_vocab(obj)
        print(
            f"[UBUNTU] decoded vocabulary entries={len(vocab)}"
        )
        for token_id in sorted(vocab)[:30]:
            print(
                f"  {token_id}: {vocab[token_id]!r}"
            )

    if dataset is not None:
        pairs=extract_ubuntu_relational(dataset,vocab)
        print(
            f"[UBUNTU] relational rows={len(pairs)}"
        )
        for pair in pairs[:args.sample_pairs]:
            print(
                f"  row={pair['row_index']} "
                f"context_ids={pair['context_ids'][:20]} "
                f"response_ids={pair['response_ids'][:20]}"
            )
            print(
                f"    context={pair['user']!r}"
            )
            print(
                f"    response={pair['reply']!r}"
            )

    if w_path.exists():
        print(
            f"[UBUNTU] W.pkl "
            f"{w_path.stat().st_size/1024/1024:.1f}MB"
        )
        try:
            obj=load_pickle_robust(w_path)
            print(
                f"  python_type={type(obj).__name__}"
            )
            print(
                f"  shape={getattr(obj,'shape',None)}"
            )
            print(
                f"  dtype={getattr(obj,'dtype',None)}"
            )
        except Exception as exc:
            print(
                f"  load_error={type(exc).__name__}: {exc}"
            )


if __name__=="__main__":
    main()
