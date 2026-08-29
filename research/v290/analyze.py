
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("--top",type=int,default=20)
    args=p.parse_args()

    payload=json.loads(
        args.input.read_text(
            encoding="utf-8"
        )
    )

    print(
        "=== V290 ANTI-SHORTCUT SEARCH ==="
    )
    print(
        "strategy space:",
        payload["space_size"],
    )

    for i,row in enumerate(
        payload["top"][:args.top],
        1,
    ):
        print(
            f"{i:2d}. {row['name']}"
        )
        print(
            f"    eval={row['eval_accuracy']:.3f} "
            f"memory_drop={row['causal_memory_drop']:.3f} "
            f"swap={row['causal_swap_expected']:.3f} "
            f"swap_change={row['causal_swap_change']:.3f}"
        )


if __name__=="__main__":
    main()
