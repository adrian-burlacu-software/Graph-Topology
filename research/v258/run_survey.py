
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent

REQUIRED=(
    "battery.py",
    "preflight.py",
    "architecture_preflight.py",
    "experiment_core.py",
    "survey.py",
    "dataset.py",
    "model.py",
    "state.py",
)

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

def value_after(argv,flag):
    if flag not in argv:
        return None

    i=argv.index(flag)

    if i+1>=len(argv):
        raise SystemExit(
            f"Missing value after {flag}"
        )

    return argv[i+1]

if __name__=="__main__":
    print(
        "=== V258 BATTERY -> OPTIONAL PARALLEL PREFLIGHT -> SERIAL SURVEY ===",
        flush=True,
    )

    missing=[
        x for x in REQUIRED
        if not (HERE/x).exists()
    ]

    if missing:
        print(
            "PACKAGE INCOMPLETE — missing: "
            + ", ".join(missing),
            flush=True,
        )
        raise SystemExit(2)

    argv=sys.argv[1:]
    skip_architecture_preflight=(
        "--skip-architecture-preflight" in argv
    )

    rc=run([
        sys.executable,
        str(HERE/"battery.py"),
    ]).returncode

    if rc:
        raise SystemExit(rc)

    pf=[
        sys.executable,
        str(HERE/"preflight.py"),
    ]

    for flag in ("--samples","--seed"):
        value=value_after(argv,flag)
        if value is not None:
            pf += [flag,value]

    rc=run(pf).returncode
    if rc:
        print(
            "RANDOM TASK PREFLIGHT FAILED — STOPPING.",
            flush=True,
        )
        raise SystemExit(rc)

    if skip_architecture_preflight:
        print(
            "SKIP: architecture preflight "
            "(--skip-architecture-preflight)",
            flush=True,
        )
    else:
        af=[
            sys.executable,
            str(HERE/"architecture_preflight.py"),
        ]

        # IMPORTANT: these options go ONLY to the parallel preflight.
        for flag in (
            "--samples",
            "--seed",
            "--device",
            "--steps",
            "--lr",
            "--parallelism",
            "--hidden-size",
            "--heads",
            "--depth",
            "--topk",
            "--terminal-weight",
            "--batch-size",
        ):
            value=value_after(argv,flag)
            if value is not None:
                target=(
                    "--preflight-steps"
                    if flag=="--steps"
                    else flag
                )
                af += [target,value]

        rc=run(af).returncode
        if rc:
            print(
                "PARALLEL SHARED-ENGINE PREFLIGHT FAILED — SURVEY WILL NOT RUN.",
                flush=True,
            )
            raise SystemExit(rc)


    print(
        "ALL PREFLIGHTS PASSED — launching SERIAL V258 survey.",
        flush=True,
    )

    survey_args=[]

    scalar_flags={
        "--samples",
        "--epochs",
        "--seed",
        "--lr",
        "--hidden-size",
        "--heads",
        "--depth",
        "--topk",
        "--terminal-weight",
        "--batch-size",
        "--log-every",
        "--output-dir",
        "--dataset-output",
        "--parallelism",
        "--device",
    }

    i=0
    while i<len(argv):
        token=argv[i]

        if token=="--skip-architecture-preflight":
            i+=1
            continue

        if token=="--steps":
            i+=2
            continue

        if token in scalar_flags:
            if i+1>=len(argv):
                raise SystemExit(
                    f"Missing value after {token}"
                )
            survey_args += [token,argv[i+1]]
            i+=2
            continue

        if token in ("--architectures","--horizons"):
            survey_args.append(token)
            i+=1
            while (
                i<len(argv)
                and not argv[i].startswith("--")
            ):
                survey_args.append(argv[i])
                i+=1
            continue

        survey_args.append(token)
        i+=1

    # Default actual survey parallelism is 2. User may override explicitly.
    if "--parallelism" not in survey_args:
        survey_args += [
            "--parallelism",
            "2",
        ]

    print(
        "PRE-FLIGHTS PASSED — launching V258 survey "
        "with parallelism=2 default.",
        flush=True,
    )

    raise SystemExit(
        run([
            sys.executable,
            str(HERE/"survey.py"),
            *survey_args,
        ]).returncode
    )
