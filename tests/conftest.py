"""Make the source checkout importable for CPU regression tests."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cond-cdvae"))
os.environ.setdefault("PROJECT_ROOT", str(ROOT / "cond-cdvae"))
