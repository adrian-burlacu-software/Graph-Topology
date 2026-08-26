from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve()


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(HERE / "train.py"),
            "--epochs",
            "2",
            "--samples",
            "1400",
            "--batch-size",
            "16",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(HERE / "evaluate.py"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
