from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CLI = Path(__file__).resolve().parent / "assistant_cli.py"


def main():
    proc = subprocess.run(
        [sys.executable, str(CLI), "--no-trace"],
        input="/quit\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "V516 cognitive bridge assistant ready." in proc.stdout, proc.stdout
    assert "Commands: /new  /status  /freeze  /unfreeze  /quit" in proc.stdout, proc.stdout
    print("V516 CLI smoke test: PASS")


if __name__ == "__main__":
    main()
