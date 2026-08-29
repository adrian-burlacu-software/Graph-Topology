
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def run(cmd):
    print(
        "\n>>> "+" ".join(cmd),
        flush=True,
    )
    return subprocess.run(
        cmd,
        cwd=HERE,
        check=False,
    )


if __name__=="__main__":
    argv=sys.argv[1:]

    # Keep task construction gated even when the expensive learning preflight
    # is skipped.
    pf=[
        sys.executable,
        str(HERE/"preflight.py"),
    ]

    for flag in ("--pairs-per-probe","--seed"):
        if flag in argv:
            i=argv.index(flag)
            pf += [flag,argv[i+1]]

    rc=run(pf).returncode
    if rc:
        print(
            "CAUSAL TASK PREFLIGHT FAILED — STOPPING.",
            flush=True,
        )
        raise SystemExit(rc)

    print(
        "CAUSAL PREFLIGHT PASSED — launching V284 compact benchmark.",
        flush=True,
    )

    raise SystemExit(
        run([
            sys.executable,
            str(HERE/"survey.py"),
            *argv,
        ]).returncode
    )
