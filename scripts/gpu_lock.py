"""Run a command while holding a machine-wide lock, so only one real-model process uses the GPU/RAM at a time.

uv run python scripts/gpu_lock.py -- uv run pytest -m model
"""
import fcntl
import subprocess
import sys
import tempfile
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "qal-scribe-gpu.lock"

cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
if not cmd:
    sys.exit("usage: gpu_lock.py -- <command>")
with open(LOCK, "w") as f:
    print(f"waiting for {LOCK}...", file=sys.stderr, flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    sys.exit(subprocess.call(cmd))
