
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

mods=[
    "semantic_memory",
    "semantic_architecture",
    "babylm_grammar",
    "roundtrip_cognitive",
    "roundtrip_benchmark",
    "real_roundtrip_benchmark",
]
for name in mods:
    __import__(name)

from real_roundtrip_benchmark import main
print("V386 local-module preflight: PASS")

if __name__=="__main__":
    import subprocess
    import sys as _sys
    raise SystemExit(
        subprocess.call(
            [_sys.executable, str(ROOT/"real_roundtrip_benchmark.py"), "--smoke"]
        )
    )
