
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS=(
    "normal_accuracy",
    "pair_discrimination_rate",
    "normal_vs_zero_drop",
    "workspace_swap_directional_rate",
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    args=p.parse_args()

    raw=json.loads(
        args.input.read_text(
            encoding="utf-8"
        )
    )
    rows=raw if isinstance(raw,list) else raw["results"]

    rows=[
        r for r in rows
        if int(r["horizon"])==4
    ]

    groups=defaultdict(list)
    for row in rows:
        groups[row["architecture"]].append(
            row["metrics"]
        )

    print("=== V288 H4-ONLY SUMMARY ===")

    for arch,items in sorted(groups.items()):
        print(
            f"{arch} H4 n={len(items)}"
        )

        for field in FIELDS:
            values=[
                float(m[field])
                for m in items
            ]

            mean=statistics.mean(values)
            sd=(
                statistics.stdev(values)
                if len(values)>1
                else 0.0
            )

            print(
                f"  {field}: "
                f"mean={mean:.3f} "
                f"std={sd:.3f} "
                f"min={min(values):.3f} "
                f"max={max(values):.3f}"
            )

        print()

    print(
        "Primary decision: H4 causal metrics, not raw accuracy alone."
    )


if __name__=="__main__":
    main()
