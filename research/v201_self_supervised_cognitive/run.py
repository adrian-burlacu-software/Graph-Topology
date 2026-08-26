from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
TRAIN = HERE / "train_self_supervised.py"
EVAL = HERE / "evaluate_self_supervised.py"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(TRAIN),
            "--epochs",
            "2",
            "--samples",
            "1024",
            "--batch-size",
            "16",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(EVAL),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
