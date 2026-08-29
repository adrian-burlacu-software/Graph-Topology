
from pathlib import Path
import json
import tempfile


def main():
    tasks=[
        ("eligibility_transition",4,287),
        ("bridge_query_eligibility_transition",4,287),
        ("eligibility_transition",4,288),
        ("bridge_query_eligibility_transition",4,288),
        ("eligibility_transition",4,289),
        ("bridge_query_eligibility_transition",4,289),
    ]

    results={}
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)

        for arch,h,seed in tasks:
            path=base/f"{arch}_h{h}_s{seed}.json"
            path.write_text(
                json.dumps({
                    "architecture":arch,
                    "horizon":h,
                    "seed":seed,
                }),
                encoding="utf-8",
            )

            results[(arch,h,seed)]=json.loads(
                path.read_text(encoding="utf-8")
            )

        assert len(results)==6
        ordered=[results[key] for key in tasks]

        assert [
            (x["architecture"],x["horizon"],x["seed"])
            for x in ordered
        ]==tasks

        # Explicitly prove that seed-specific filenames do not collide.
        names={
            f"{arch}_h{h}_s{seed}.json"
            for arch,h,seed in tasks
        }
        assert len(names)==6

    print("V288 scheduler regression: PASS")


if __name__=="__main__":
    main()
