"""Put src/ at the **front** of sys.path so that `pytest tests/` works without `pip install -e .`.

Why the front: if a package of the same name is already installed, that one wins and the tests
exercise the installed copy rather than the files next to them. Passing would then say nothing
about this package. The tests always read **the source sitting beside them**.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
