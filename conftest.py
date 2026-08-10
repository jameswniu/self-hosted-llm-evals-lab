"""Put the pipeline directories on sys.path so tests can import them directly.

The modules in this repo are scripts run by the Makefile, not an installed
package, so there is nothing to pip install. This keeps the test suite runnable
on a clean checkout with no packaging ceremony.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("ablation", "docs", "eval_runner", "validate"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
