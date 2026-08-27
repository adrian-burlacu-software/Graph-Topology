from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    raise SystemExit(subprocess.run(
        [sys.executable, str(HERE / "train.py"), *sys.argv[1:]]
    ).returncode)
