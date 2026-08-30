
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

CONCEPTNET_57=(
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=r".\data\conceptnet-assertions-5.7.0.csv.gz")
    args=ap.parse_args()

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)

    if out.exists():
        print(f"[CONCEPTNET] exists: {out}",flush=True)
        return

    print(f"[CONCEPTNET] downloading {CONCEPTNET_57}",flush=True)

    def report(block,total,done):
        if total:
            print(
                f"\r  {done/total*100:6.2f}% "
                f"{done/1024/1024:8.1f}MB",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        CONCEPTNET_57,
        str(out),
        reporthook=report,
    )
    print("\n[CONCEPTNET] complete",flush=True)

if __name__=="__main__":
    main()
