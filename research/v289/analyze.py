
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("--top",type=int,default=12)
    args=p.parse_args()

    payload=json.loads(
        args.input.read_text(encoding="utf-8")
    )

    print("=== V289 SEARCH RESULTS ===")
    print(
        "combinatorial space:",
        payload["space_size"],
    )
    print(
        "seeds:",
        payload["seeds"],
        "horizons:",
        payload["horizons"],
    )
    print()

    for rank,row in enumerate(
        payload["top"][:args.top],
        1,
    ):
        s=row["search"]
        c=row["causal"]

        print(
            f"{rank:2d} {s['name']}"
        )
        print(
            f"   search accuracy       = "
            f"{s['accuracy']:.3f}"
        )
        print(
            f"   H2/H4 task accuracy   = "
            f"{s['by_horizon'].get('4',{}).get('accuracy',0):.3f}"
        )
        print(
            f"   causal normal         = "
            f"{c['normal_accuracy']:.3f}"
        )
        print(
            f"   memory ablation drop  = "
            f"{c['memory_ablation_drop']:.3f}"
        )
        print(
            f"   swap correct          = "
            f"{c['swap_correct_rate']:.3f}"
        )
        print()


if __name__=="__main__":
    main()
