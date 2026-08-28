
from pathlib import Path
import subprocess,sys

HERE=Path(__file__).resolve().parent

def run(cmd):
    print("\n>>> "+" ".join(cmd),flush=True)
    return subprocess.run(
        cmd,
        cwd=HERE,
        check=False,
    )

if __name__=="__main__":
    argv=sys.argv[1:]

    pf=[
        sys.executable,
        str(HERE/"preflight.py"),
    ]

    for flag in (
        "--pairs-per-horizon",
        "--seed",
    ):
        if flag in argv:
            i=argv.index(flag)
            pf += [flag,argv[i+1]]

    rc=run(pf).returncode

    if rc:
        print(
            "PREFLIGHT FAILED — STOPPING.",
            flush=True,
        )
        raise SystemExit(rc)

    print(
        "PREFLIGHT PASS — launching V277 fault-isolation run.",
        flush=True,
    )

    raise SystemExit(
        run([
            sys.executable,
            str(HERE/"isolated_memory.py"),
            *argv,
        ]).returncode
    )
