import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke():
    r = subprocess.run([sys.executable, str(ROOT / "mujoco_env" / "smoke_test.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok:")
