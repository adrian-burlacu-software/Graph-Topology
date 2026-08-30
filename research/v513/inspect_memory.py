
from __future__ import annotations

import argparse
import sqlite3


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--memory",
        default=r".\results\full_semantic_memory.sqlite",
    )
    args=ap.parse_args()

    con=sqlite3.connect(args.memory)

    print("[MEMORY]")
    for table in (
        "sources","concepts","facts","utterances",
        "ubuntu_pairs","ubuntu_pair_tokens","ubuntu_token_vocab",
        "udgum_sentences","udgum_tokens",
        "verbnet_classes","verbnet_members",
        "verbnet_roles","verbnet_frames",
    ):
        try:
            n=con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        except sqlite3.Error:
            n="n/a"
        print(f"  {table}: {n}")

    print("\n[DATASETS]")
    for row in con.execute(
        "SELECT dataset,COUNT(*) FROM sources GROUP BY dataset ORDER BY dataset"
    ):
        print(f"  {row[0]}: {row[1]} source records")

    print("\n[FACT TYPES]")
    for row in con.execute(
        "SELECT fact_type,COUNT(*) FROM facts GROUP BY fact_type"
    ):
        print(f"  {row[0]}: {row[1]}")

    con.close()


if __name__=="__main__":
    main()
