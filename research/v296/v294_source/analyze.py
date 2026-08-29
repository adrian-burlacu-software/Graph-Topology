
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
            encoding="utf-8"
        )
    )

    print("=== V294 ARCHITECTURE DISCOVERY ===")
    print(
        "space=",data["space_size"],
        "tasks=",",".join(data["tasks"]),
        "horizon=",data["horizon"],
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
            f"gain={row['learning_gain']:+.3f} "
            f"synergy={row['architecture_synergy']:.3f}"
        )
        print(
            "    drops:",
            " ".join(
                f"{k}={v:+.3f}"
                for k,v in row["ablation_drop"].items()
            )
        )


if __name__=="__main__":
    main()
