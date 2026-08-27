
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
raise SystemExit(subprocess.run([sys.executable,str(HERE/"battery.py"),*sys.argv[1:]]).returncode)
