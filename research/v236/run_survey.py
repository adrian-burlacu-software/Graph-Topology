
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
if __name__=="__main__":
    print("=== V236 BATTERY -> ARCHITECTURAL DECISION SURVEY ===",flush=True)
    rc=subprocess.run([sys.executable,str(HERE/"battery.py")],cwd=HERE).returncode
    if rc!=0:
        print("BATTERY FAILED — SURVEY WILL NOT RUN.",flush=True)
        raise SystemExit(rc)
    print("BATTERY PASSED — launching V236 survey.",flush=True)
    raise SystemExit(
        subprocess.run(
            [sys.executable,str(HERE/"survey.py"),*sys.argv[1:]],
            cwd=HERE,
        ).returncode
    )
