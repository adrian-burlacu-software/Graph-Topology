
from __future__ import annotations
import argparse,json,statistics
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
    raw=json.loads(args.input.read_text(encoding="utf-8"))
    rows=raw if isinstance(raw,list) else raw["results"]
    groups=defaultdict(list)
    for row in rows:
        groups[
            (row["architecture"],int(row["horizon"]))
        ].append(row["metrics"])
    print("=== V287 SMALL DECISIVE SUMMARY ===")
    for (arch,h),items in sorted(groups.items()):
        print(f"{arch} H{h} n={len(items)}")
        for field in FIELDS:
            values=[float(x[field]) for x in items]
            mean=statistics.mean(values)
            sd=statistics.stdev(values) if len(values)>1 else 0.0
            print(
                f"  {field}: mean={mean:.3f} "
                f"std={sd:.3f} min={min(values):.3f} "
                f"max={max(values):.3f}"
            )
        print()

if __name__=="__main__":
    main()
