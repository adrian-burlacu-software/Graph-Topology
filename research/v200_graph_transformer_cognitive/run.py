from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
TRAIN = HERE / "train.py"
EVALUATE = HERE / "evaluate.py"


def main() -> None:
    print("=== V200 SMOKE RUN ===")
    subprocess.run(
        [sys.executable, str(TRAIN), "--epochs", "1", "--samples", "512", "--batch-size", "16"],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(EVALUATE), "--samples", "256"],
        check=True,
    )


if __name__ == "__main__":
    main()
