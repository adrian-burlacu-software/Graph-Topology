
from __future__ import annotations

import argparse
import nltk


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--all",action="store_true")
    args=ap.parse_args()

    packages=["wordnet","omw-1.4","verbnet"]
    for name in packages:
        print(f"[NLTK] downloading {name}",flush=True)
        nltk.download(name)

    print("[NLTK] semantic corpora ready",flush=True)


if __name__=="__main__":
    main()
