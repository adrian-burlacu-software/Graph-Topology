
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("--top",type=int,default=20)
    args=p.parse_args()

    data=json.loads(
        args.input.read_text(
            encoding="utf-8",
        )
    )

    print("=== V292 GRAPH COGNITION RESULTS ===")
    print(
        "space=",data["space_size"],
        "horizon=",data["horizon"],
        "episodes/sequence=",
        data["episodes_per_sequence"],
    )

    for i,row in enumerate(
        data["top"][:args.top],
        1,
    ):
        print(
            f"{i:2d}. {row['name']}"
        )
        print(
            f"    eval={row['eval_accuracy']:.3f} "
            f"first={row['eval_first_half']:.3f} "
            f"second={row['eval_second_half']:.3f} "
            f"gain={row['online_learning_gain']:+.3f} "
            f"memory_drop={row['memory_drop']:.3f}"
        )


if __name__=="__main__":
    main()
